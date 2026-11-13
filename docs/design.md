# Design — lightx2v-rs

Phase 2 design notes. Everything in "Verified" was actually run on this
CPU-only VM on 2026-11-13; numbers quoted from the papers / upstream
configs are marked as such and were not re-measured here. The wgpu
backend (§9) is implemented and reviewed but has not executed on a GPU
yet — that is stated explicitly wherever it matters.

## 1. What this is

A Rust port of two things that compose:

- **LightX2V** (ModelTC/LightX2V, Apache 2.0): 4-step distillation of
  Wan-class video diffusion models — no CFG, NVFP4/FP8 quant, sparse
  attention, block offload. Upstream claims up to 50× on RTX 5090/H100
  (not re-measured here). Supported backbones: Wan 2.1/2.2, LTX-2,
  HunyuanVideo 1.5, MiniMax H3, Self-Forcing profiles.
- **History-Forcing** (arXiv:2611.02782, Kuang et al., Zhejiang Univ +
  MSR): few-step *autoregressive* video diffusion. The student is
  distilled with a history-contrast objective — the teacher is run
  WITH vs WITHOUT informative early history and the weighted residual
  is added to the score/velocity target:

  `corrected_target = base_target + w_hist · (teacher_with_history − teacher_without_history)`

  Paper results (not re-measured): DMD student on Wan2.1-T2V-1.3B,
  60 s consistent, **VBench 85.07 @ 15 s, VBench-Long 79.82 @ 60 s**;
  OPS-D student on DreamX-World-5B (Wan2.2-TI2V-5B) for out-of-sight
  scene evolution.

The port follows the fastwan-rs / h3-rs shape: one fused `Backend` op
vocabulary, a CPU f32 reference backend as the oracle, and a causal
chunk pipeline. See §7 for lineage.

## 2. Authoritative numbers (from the upstream clones in `research/`)

From `history-forcing-upstream/dmd/long_video/config.py` +
`inference.yaml` (the authoritative source — read, not copied):

| Item | Value |
| --- | --- |
| Generation | latent_frames **252** (1005 video frames), 832×480, 16 fps, bf16, batch 1, seed 0, CRF 23 |
| MemRoPE | cache_frames **12 = sink 3 + long-EMA 1 + short-EMA 1 + recent 4 + new 3** |
| EMA | `ema_alpha_long 0.01`, `ema_alpha_short 0.1`, `ema_adaptive false` |
| Model | `num_frame_per_block 3`, `warp_denoising_step true`, `independent_first_frame false`, `context_noise 0` |
| Denoising | `denoising_step_list [1000, 750, 500, 250]`, `timestep_shift 5.0` |
| MemRoPE application | per-attention-layer cache update after each layer; KV cache holds q/k + `token_temporal_indices` + `token_spatial_indices` |

From `lightx2v-upstream/configs/wan/wan_t2v.json` (base profile):
`infer_steps 50`, `num_frames 81`, `text_len 512`, size `[480, 832]`,
`flash_attn3`, guide 6, shift 8, CFG on. From
`configs/wan22/wan_moe_t2v_distill.json` (distilled profile):
`infer_steps 4`, `distill_method dmd2`, `denoising_step_list
[1000, 750, 500, 250]`, `sample_shift 5.0`, `enable_cfg false`,
`guide [4.0, 3.0]`, `boundary_step_index 2`, `cpu_offload true`,
`offload_granularity model`, high/low-noise checkpoints.

These are encoded in `lightx2v-core::config` (`GenerationConfig`,
`MemRopeConfig`, `DistillConfig`) with the exact validation rules
from `config.py`: `cache_frames == sink + long + short + recent + 3`,
long/short == 1, EMA alphas in (0, 1], `latent_frames` a multiple of
3. `DistillConfig::checkpoint_for_step` selects the high/low-noise
checkpoint by `boundary_step_index 2`.

## 3. What is fused and why

The `Backend` trait (`lightx2v-core::backend`) mirrors fastwan's
fused vocabulary — linear / layer_norm / residual_norm / gate_residual /
head_split / attention / patchify / unpatchify / dmd_step / conv2d /
tanh_clamp — extended for this port with:

- `causal_attention` with a KV-cache handle + explicit block mask
  (`tk <= tq` over the cache's temporal indices);
- `kv_cache_append` / `kv_cache_put` / `kv_cache_memrope_update` /
  `kv_cache_temporal` / `kv_cache_kv` — the MemRoPE cache is a
  first-class backend object, not a model-side Vec, because the
  wgpu/CUDA backends will keep it on-device;
- `sparse_attention` with a `SparseSpec` (VSA-style tile table +
  gate) — the LightX2V sparse path, exercised by the CPU backend
  (coarse mean-pool attention → top-k → fine attention →
  `coarse·gate + fine`; with gate 0 and sparsity 0 it reduces to
  dense attention, which is a check case);
- `weight_fp8` / FP8 e4m3 quantize/dequantize — the LightX2V FP8
  path, dequantize-as-reference contract (NVFP4 micro-block is
  Phase 3);
- `OffloadGranularity` (None / Model / Block / Disk) + a memory
  plan fn — LightX2V's block offload, surfaced in `plan.rs` and the
  CLI so a 60 s rollout can be planned against a real VRAM budget
  before any weight is loaded.

Fusion rationale (same as fastwan/h3): the DiT is written once
against the trait, so a new backend is a new impl of ~30 ops, not a
new model; the CPU backend is the oracle every other backend is
checked against; and the ops are coarse enough that a GPU backend
can keep tensors on-device across a whole block (no per-element
round-trips through f32).

## 4. Causal pipeline + MemRoPE

`lightx2v-model::pipeline::CausalPipeline` generates 3-latent-frame
chunks, autoregressively, reusing the MemRoPE cache across chunks:

1. Fresh noise per chunk; 4 DMD steps per chunk (`warp_denoising_step`
   — each chunk is denoised fully once before moving on).
2. Per DiT layer: `forward_chunk` writes the chunk's K/V with
   `kv_cache_put`, runs the MemRoPE update (seed EMA zeros after the
   sinks on first non-sink append, graduate New → Recent past 3,
   evict the oldest Recent folding it into the EMA slots), builds the
   causal block mask once per chunk from the cache's temporal
   indices, and attends.
3. Caches persist across chunks (swapped in the workspace); the
   cross-attention prompt K/V is prepared once per generation.

**Cache-write discipline** (from fastvideo-rs `wan/causal.rs`, read
2026-11-13): a block's roped K/V is *appended when the block is new,
written over the block's own slots when the same block is run again*
(the next denoising step, then the clean-context pass). `kv_cache_put`
implements exactly that. The first implementation appended on every
step, so a 4-step chunk occupied the cache four times over and evicted
real history — the conformance test caught it (rel err 1.19 before
the fix, 7.2e-7 max abs after). The overwrite search must skip EMA
slots: they share the first non-sink frame's temporal index, and
overwriting one corrupts the memory.

**Deliberate divergence from fastvideo-rs causal**: lightx2v-rs
attends through an explicit causal block mask (`tk <= tq`) plus
MemRoPE EMA compression and denoises each chunk fully once
(History-Forcing inference); fastvideo-rs causal attends maskless to
the last `max_attention` window (local_attn_size frames, or
`sliding_window_num_frames` 21 when local_attn_size == −1) and
overwrites cache slots on re-run. Both keep sinks. Relativistic RoPE
(un-roped cache, re-rope the window per forward — fastvideo's
`rope_cache_policy`, `KvSpec::relativistic`) is a Phase 2/3 candidate
for open-ended streams; lightx2v-rs currently stores roped K and
rotates by original temporal indices (block RoPE).

## 5. Quantization

`lightx2v-core::quant`: Q8 group-128 and Q4 HQQ (lp-norm 0.7, β 10,
κ 1.01, ≤ 20 iters — fastwan semantics) weight-only, plus FP8 E4M3
(`Fp8E4M3`, `FP8_GROUP 128`, `FP8_E4M3_MAX 448.0`) with f32↔e4m3
codecs for the LightX2V FP8 path. NVFP4 (micro-block FP4 + E4M3
scales, the LightX2V headline quant) is Phase 3: the h3-rs contract
warning applies — ComfyUI single-file NVFP4 checkpoints need a key
map + unswizzled-scale variant and are not loadable yet.

## 6. Lineage: what is reused vs deliberately not copied

### fastwan-rs (template, read from `/tmp/fastwan/`)

Mirrored: workspace layout (core / cpu / model / cli), naming,
error discipline (missing key / shape mismatch = error), the fused
`Backend` trait shape, the CPU-reference-as-oracle conformance
discipline (tiny safetensors fixtures, `docs/runs`-style notes),
the mmap safetensors loader, Q8/Q4 quant semantics, the VRAM plan.
Code is reimplemented with lightx2v names, not copied verbatim.

### h3-rs (MiniMax H3 — a LightX2V backbone)

Read via ghapi 2026-11-13 (private repo, accessible). What it
contributes:

- **Variant presets** — `h3_core::variant` presets (h3, fasth3-v1/v2/
  trim/omni, pika-flex, regen-2k) with `fastvideo_inference.json`
  overrides. Mirrored as `DistillConfig` profiles in lightx2v-core.
- **AdaLN streaming** — the 13B modulation weights are never
  resident; tables are computed per generation by streaming each
  block's weight through the backend once (or loaded from an
  `adaln-tables` file). Mirrored as `step_mods` in the DiT.
- **NVFP4 contract warning** — see §5.
- **Check discipline** — 90/90 checks on a named GPU (RTX A6000);
  measured timings only, e.g. FastH3-V2 denoise 201 s at
  1344×768×124 (5.2 s), per-block 470 ms VSA / 815 ms dense CUDA.
- **Checkpoint contracts** — channel-major patch features,
  `value, gate` SwiGLU halves, per-head `norm_q`/`norm_k`, RoPE
  rotate-half over the first 96 of 128 channels, AdaLN slot order
  shift/scale/gate (attn) then (mlp), `x0 = x + σ·v`.

What is deliberately **not** copied: h3-rs has **no**
causal/autoregressive KV-cache chunked rollout — it denoises full
clips, and continuation is via Ref2VA keyframes, not a stream. So
its pipeline is not the model for lightx2v-rs's causal pipeline
(§4 follows History-Forcing + fastvideo-rs instead). Also not
copied: the audio VAE, the Qwen3-VL conditioner (Phase 3+), and the
SwiGLU/rotate-half-RoPE specifics (the Wan DiT differs). H3 as a
LightX2V backbone (upstream `configs/minimax_h3*`,
`streamrender_h3`, `taomate_h3`) is a Phase 3 variant.

### fastvideo-rs (upstream of fastwan-rs)

Read via ghapi 2026-11-13 (public, main tip 5b0d9a34). What
lightx2v-rs inherits:

- **mmap safetensors loader** (fastvideo-loader — `lazy.rs`,
  `prefetch.rs`, `stream.rs`) → `weights.rs` (`SafetensorsDir`,
  `Synthetic`, manifests).
- **FP8 ops** (fastvideo-ops E4M3) → `quant.rs` `Fp8E4M3`.
- **Golden/parity test discipline** (gold-check/gold-run workflows,
  `artifacts/perf/` per-GPU traces) → conformance tests +
  `docs/runs`-style notes (only for things actually run).
- **Cloudflare edge Worker + Durable Object crate**
  (fastvideo-edge) → Phase 3 serve idea.
- **Causal discipline** (`wan/causal.rs`, `causal_transformer.py`,
  `causal_denoising.py`) → `kv_cache_put`, §4.

Deliberate divergence: fastwan-rs / lightx2v-rs scope to **one
model family + consumer GPUs + a fused `Backend` trait**; fastvideo-rs
is datacenter CUDA-first (cudarc primary generate path, Runpod RTX
PRO 6000, sequence parallel `--num-gpus N`, full serve/autoscale/
deploy stack, protocol/trace crates). Candle is fastvideo-rs's frozen
behavioral oracle; Luminal is frozen.

## 7. Verified (actually run, 2026-11-13, CPU-only VM)

- `cargo build --workspace --offline`: passes. `cargo test
  --workspace --offline`: **37 passed, 0 failed** (28 core, 2 model,
  4 conformance, 3 wgpu — the wgpu 3 skip: no adapter on this VM).
  `cargo clippy --workspace --offline -- -D warnings`:
  clean. `cargo fmt --check`: clean. CPU-only build
  (`cargo build -p lightx2v-cli --no-default-features`) passes too.
- `lightx2v check --backend cpu`: **11/11 passed** (linear, layer_norm,
  attention, causal_attention_mask, sparse==dense@0-sparsity-gate-0,
  patchify/unpatchify roundtrip, dmd_step, quant q8/q4/fp8,
  rope origin identity, memrope cache bounded + sinks + ema,
  history-contrast residual). `check --backend wgpu` prints a clean
  skip (`no WebGPU adapter answered`) and exits 0; `check --backend
  auto` falls back to cpu, 11/11.
- Conformance (`crates/lightx2v-model/tests/conformance.rs`):
  2-chunk tiny-DiT rollout vs an independent f32 reference
  (`tests/reference.rs`), max abs error **7.2e-7**; KV-cache test
  bounds the cache at 12 frames and keeps the sinks; TAEHV decode
  test decodes 3 latent frames → 9 video frames at 64×64,
  deterministic on repeat decode, and bit-identical whether decoded
  as one 6-frame chunk or 4+2 chunks (MemBlock memory carried across
  the boundary); quantize flag parses. The test asserts
  max-abs < 1e-5, not relative error: relative error blows up on
  near-zero elements (9.6e-5 rel at 7.2e-7 abs is f32
  accumulation-order noise, not a semantic difference).
- `lightx2v plan` (81 frames, 16 GiB): peak **7638.8 MiB, fits**.
  `plan --frames 1005` (60 s demo): peak **17992.6 MiB, does not
  fit 16 GiB**; `--offload block` keeps 189,199,572 of
  2,837,993,600 weight bytes resident. MemRoPE cache bounded at
  12 frames in both.
- `lightx2v bench --layers 1`: tiny DiT, 1 chunk × 4 DMD steps,
  **0.203 s** total (load 0.001 s, prompt 0.001 s, denoise 0.202 s).
- `lightx2v generate --synthetic --out clip`: 3 latent
  frames (9 video frames), 768 elems, **0.28 s**; writes `clip.latent`
  (raw f32) **and** `clip.rgb` (TAEHV-decoded, 9 frames 64×64,
  110,592 bytes). `generate` without `--synthetic` errors cleanly
  naming the missing model dir / UMT5 blocker.

Two real bugs found and fixed during conformance (both caught by
tests, not by inspection):

1. **CPU `head_split` RMS-normalised `v`.** With `rms_w = None`
   it still applied RMS across the row; Diffusers leaves `v` raw.
   Divergence 0.63 absolute. Fix: RMS only when a weight is given.
2. **`unpatchify` used a different feature order than `patchify`.**
   Patchify is channel-major (`cc·p² + kh·p + kw`); unpatchify
   read spatial-major (`(kh·p+kw)·c + cc`). The DiT conformance
   test could not catch it — the independent reference had the
   same wrong assumption — but the `check` roundtrip did
   (max abs 4.148 before, 0.0 after). Fix: channel-major
   everywhere, matching the checkpoint contract.

## 8. What is next (Phase 3)

- **Phase 3**: CUDA backend, NVFP4 micro-block quant, real DiT
  weights (`--model-dir` generation end-to-end — the blocker is UMT5
  text encoding, not the DiT or the decoder), UMT5 text encoding,
  relativistic RoPE for open-ended streams, H3 backbone variant,
  Cloudflare edge serve (fastvideo-edge pattern), sequence-parallel /
  multi-GPU plan, `docs/runs/` measurements on named hardware. No
  speed claim until measured.
- Never deploy or quote the paper's VBench numbers as measured
  here — they are the upstream paper's, cited in §1.

## 9. Phase 2: wgpu backend + TAEHV decode

### What fused and why

Phase 2 adds no new `Backend` ops: the whole trait from §3 runs on
wgpu, including the LightX2V extensions — `causal_attention` with a
block mask, `kv_cache_put` (append-or-overwrite), `sparse_attention`
(VSA), and `kv_cache_memrope_update`. One WGSL kernel per op, the
same fusion rationale as §3: the DiT is written once against the
trait, so the wgpu backend is a new impl of ~30 ops, not a new
model, and tensors stay on-device across a whole block.

### Buffer / KV-cache layout

- Activations are f32 buffers; weights are f16 pairs packed in `u32`
  and unpacked with core `unpack2x16float` (no `shader-f16`
  dependency). A weight whose values do not fit f16 (or whose row
  length is odd) stays f32 — never silently clipped. Q8/Q4 weights
  stay packed and are dequantized in the shader; FP8 weights are
  dequantized on the host into the same f16-packed form. `wgpu`
  validates shaders at pipeline creation (runtime, only when a
  device exists), so shader syntax errors do not surface on a
  headless machine — every shader was re-read against the CPU oracle
  line by line instead.
- Dispatches are recorded into one command encoder and submitted in
  batches of 64; a host write or read flushes first, so queue
  ordering always matches program order. Params travel as a fresh
  UNIFORM buffer per dispatch (padded to a multiple of 4 u32s).
- KV cache: one layer's K/V on-device as `[rows, heads·d]`
  row-major over tokens — the CPU oracle's layout, so `kv_cache_kv`
  is a transpose away from attention's `[heads, rows, d]` input. The
  temporal index and `SlotKind` of every row live on the host (small
  `Vec`s); the device never sees them. `kv_cache_put` decides
  append-vs-overwrite on the host and issues one `kv_pack` kernel
  either way. `kv_cache_memrope_update` runs the same metadata
  simulation as the CPU oracle and executes it with two kernels:
  `ema_fold` (in place, one frame into an EMA slot, via a temp
  buffer so no buffer is bound read-write and read at once) and
  `kv_compact` (gather rows by an index map, zero-filling freshly
  seeded EMA frames).

### Shader list (`crates/lightx2v-wgpu/src/shaders/`)

| Shader | Ops |
| --- | --- |
| `common.wgsl` | `gelu_tanh` / `tanh_safe` (exp form; WGSL `tanh` overflows to NaN on some drivers), `wg_index` |
| `linear.wgsl` | 64×64 tiled GEMM, `{W_TYPE}`/`{W_LOAD}` templated per storage (f32 / f16-packed / Q8 / Q4); row offsets for chunked FFN |
| `norm.wgsl` | mode 0 plain LN, mode 1 residual+LN in place on `h` (the norm reads the unrounded sum, as the CPU does); affine or AdaLN-mod |
| `head_split.wgsl` | one workgroup per token; RMS only when a weight is given (v passes through raw); interleaved-pair RoPE; scatter to `[heads, rows, d]` |
| `flash.wgsl` | one-row-per-thread online-softmax attention, three modes: 0 dense, 1 VSA fine stage (selected tiles + `coarse·gate`), 2 causal masked (`mask[qi·sk+ki]`). Invalid keys score −1e30 and are skipped; an all-masked row outputs 0 (the CPU reference would produce NaN there — the DiT never has an all-masked row, every query sees the sink frames) |
| `conv.wgsl` | implicit GEMM, NCHW stride 1 pad k/2; concat + nearest-×2 upsample by address in the tile loader; epilogues none / relu / AddRelu / Prelu |
| `elementwise.wgsl` | ops 0 gate_residual, 1 dmd_step, 2 tanh_clamp, 3 patchify, 4 unpatchify (channel-major — fastwan's op 4 was channel-last and is corrected here), 5 kv_pack, 6 kv_unpack, 7 ema_fold |
| `vsa_pool.wgsl` | one workgroup per (tile, head); mean-pool Q/K/V over the tile's real tokens |
| `vsa_coarse.wgsl` | one workgroup per (query tile, head); coarse scores + softmax + coarse out, top-k by bitwise threshold search on order-preserving float keys (ties to the lower index, as the CPU's sort) |
| `kv_compact.wgsl` | gather rows by a u32 map, PAD = zero row |

Phase 2 deliberately uses the simple one-row-per-thread flash kernel
(fastwan's fallback shape), not the register-tiled f16 variant: the
tiled kernel is Phase 3 perf work, not a correctness gap. Likewise
there is no profiler and no f16-math kernel yet.

### TAEHV decode

`TaeHv::decode` (in `crates/lightx2v-model/src/taehv.rs`) ports
fastwan-model's decoder with the same three exact rewrites: TGrow
(1×1) before the upsample, upsample fused into the next conv,
concat fused into the MemBlock's first conv. Latents decode in
4-latent-frame chunks; each MemBlock carries its last input frame
across the boundary (the reference's sequential memory), and the
first 3 video frames are trimmed as priming. `decode` takes the
video dimensions explicitly — the toy runs denoise under the demo
profile while their video is 64×64. Conv weights are plain backend
weights; a checkpoint storing a conv weight quantized is dequantized
on the host at load (`conv2d` has no quantized path).

### What is verified vs what still needs a real GPU

Verified on this VM (§7): the wgpu crate builds offline and passes
clippy/fmt; the 3 wgpu conformance tests run and take the clean skip
path; the TAEHV decode is verified on CPU (shape, determinism,
chunk-invariance); the CLI's `--backend` selection and skip paths
are verified end to end (cpu 11/11, wgpu skip exit 0, auto falls
back to cpu, CPU-only build skips wgpu with the feature reason).

**Not verified: no wgpu kernel has executed.** This VM has no
Vulkan driver (`/usr/share/vulkan/icd.d/` does not exist, no
lavapipe `.so` anywhere, `apt install mesa-vulkan-drivers` has no
installation candidate — apt lists are unreachable). First run on a
real GPU: `lightx2v check --backend wgpu` (expect 11/11) and
`cargo test -p lightx2v-wgpu` (tiny-DiT rollout vs CPU < 1e-5 with
f32 weights; TAEHV decode vs CPU < 1e-3 with f32 weights, and the
f16-packed decode error printed and bounded at 5e-2). The wgpu
tests force `allow_f16_weights = false` for the strict comparisons
so a failure at 1e-5 is a kernel bug, not f16 rounding — if a check
fails there, investigate, don't loosen the tolerance.

## 10. Prompt Studio (Phase 2)

The lightx2v equivalent of fastwan's Prompt Studio — a local engine plus
a single-file web UI, NOT a hosted-API studio and NOT an LTX port. The
serve structure (engine thread + mpsc job queue + shared job table,
`GET /` + `/api/health` + `/api/gallery` + `POST /api/generate` +
`/api/jobs/<id>` + `/frames` binary, JSON-only + Origin guard on POST,
MAX_QUEUE 3, KEEP_JOBS 8, CORS for exactly one `--allow-origin`, engine
panic → job error + health) mirrors fastwan-rs's `serve.rs`, reimplemented
against the lightx2v pipeline. The differences are the API surface and
the long-video chunk progress, which fastwan has no equivalent of.

### Architecture

```text
lightx2v serve [--synthetic | --model-dir <dir> [--gallery <dir>]]
  [--addr 127.0.0.1:7860] [--allow-origin <url>] [--backend cpu|wgpu|auto]
    engine thread: owns Backend + resident DiT/TAEHV, runs jobs one at a time
    HTTP thread (tiny_http): serves web/studio.html + the JSON/binary API
lightx2v quantize --transformer <dir> --weights int4-hqq-g64|native --out <file>
  crates/lightx2v-core/src/export.rs: block-ordered safetensors writer
lightx2v generate --model-dir <dir> --embeds <file> [--weights <fmt>]
  real generation from precomputed prompt_embeds (UMT5 is Phase 3)
  web/studio.html: prompt box, size/length/seed, backend selector
    [In this browser (WebGPU) / Local server], long-video toggle with
    per-chunk progress, canvas playback as frames arrive, play/scrub,
    save WebM (MediaRecorder) + PNG, gallery picker, synthetic badge
  crates/lightx2v-web: wasm-bindgen API (createEngine("lightx2v"|"tiny"),
    addBlock / finishDit / loadTae / generate with prepared/step/chunk/
    frames events) on the WebGPU backend; DitBuilder loads one block at a
    time; WgpuBackend::download_async is the only readback (the sync
    download would hang the tab)
```

The engine denoises with `CausalPipeline::denoise_with`, whose
`Event::{Prepared, Step, Chunk, Latent}` drive the job's stage, per-step
timings and `chunks_done`/`total_chunks`; decode is progressive through
`TaeHv::decode_chunk` (4 latent frames at a time, `DecodeState` carrying
the MemBlock memory across boundaries), so `GET /frames?from=N` returns
frames as they land instead of only at the end. For the 60 s demo that
is 84 chunks: the page's chunk bar moves 84 times and the first frames
play while chunk 2 is still denoising. The browser engine mirrors the
same rollout in `lightx2v-web::run` (it cannot call `denoise_with`:
that path blocks on `download`/`sync`, which hang on WebGPU) — same
`patch_embed` → `forward_chunk` → `unpatchify` → `dmd_step` per chunk,
with a `download_async` on the fence buffer after every DMD step for
honest timing, and progressive `launch_chunk` decode as each chunk's
latent lands.

### Weights: export, streaming, hosting

The browser runs the real model — synthetic is only the dev fallback.
The chain, mirroring fastwan-rs exactly:

1. **Export** — `lightx2v quantize --transformer <dir>/transformer
   --weights int4-hqq-g64 --out web-weights/lightx2v-1.3b-f4.safetensors`
   (or `--weights native` for the f16 file). `export::quantize_dit`
   reads the HF transformer dir through `SafetensorsDir`, quantizes
   block linears per `WeightQuant::for_key` (FFN keys get the format,
   attention/gate keys get `attn()` — Int8 when `-ffn`, else the same),
   and writes with `export::write` + `export::metadata`: head tensors
   first, then blocks in order, each tensor's data contiguous — the
   file is **block-ordered on write**. f4 ≈ 0.88 GB, f16 ≈ 2.98 GB,
   TAEHV ≈ 23 MB (`web-weights/source/taew2_1.safetensors`, fetched
   from fastwan's public R2; the HF `madebyollin/taehv` repo is gated).
   Weights live under `web-weights/` (git-ignored, never committed).
   The source transformer is the fp32
   `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` checkpoint (5.96 GB
   single safetensors file) — lightx2v runs its own 4-step DMD
   schedule `[1000, 750, 500, 250]` (shift 5.0) over these 3-step-
   distilled weights; the true History-Forcing 4-step checkpoint is
   `xzxxntxdy/History-Forcing` on HF, the alternative source.
2. **Streaming** — the page fetches `manifest.json` from the weights
   base URL, reads the safetensors header with two Range requests,
   groups tensors into head + `blocks.N`, and **refuses the file if
   any group's bytes are not contiguous** (`readGroups` throws
   "not block-ordered (re-export it with lightx2v quantize)"). Each
   group is fetched with one Range request, cached in the Cache API
   (`lightx2v-weights-v1`, keyed by file + size + group), sliced into
   a standalone safetensors blob (`groupBlob`), and handed to
   `Engine.addBlock`; the head group goes to `finishDit`. The TAEHV
   loads whole (`loadTae`). `?weights=<base URL>` overrides the base
   baked in by `build-site.py --weights`.
3. **Hosting (parent runs this; do not upload from this task)** —
   `scripts/deploy-pages.sh --weights <base URL>` builds `web/dist/`
   (page + wasm pkg + prompts + `manifest.json`, never weights). The
   weights upload, once, from the export dir:
   ```sh
   # R2 bucket `lightx2v-weights`, public dev URL
   # https://pub-<id>.r2.dev/v1 — replace with the real one.
   # CORS on the bucket must be set explicitly — a public r2.dev URL
   # serves NO Access-Control-Allow-Origin by default (proven 2026-11-13:
   # the page reported "Cannot reach the weights" until the policy below
   # was set; the GET then returned ACAO: * and the Range preflight 204).
   #   wrangler r2 bucket cors set lightx2v-weights --file cors.json
   # cors.json (wrangler shape):
   # {"rules": [{"allowed": {"origins": ["*"], "methods": ["GET", "HEAD"],
   #   "headers": ["Range"]}, "exposeHeaders": ["Content-Length",
   #   "Content-Range", "Accept-Ranges", "ETag"], "maxAgeSeconds": 86400}]}
   # NOTE: `wrangler r2 object put` refuses files >300 MiB (f4 837 MB,
   # f16 2.98 GB) — those need S3 multipart upload (boto3 with an R2 API
   # token; the stored Cloudflare credential cannot create one, 403).
   # And `wrangler r2 bucket cors set` needs the Cloudflare credential,
   # not the object-scoped R2 token (PutBucketCors → AccessDenied).
   wrangler r2 object put lightx2v-weights/v1/lightx2v-1.3b-f4.safetensors \
     --file web-weights/lightx2v-1.3b-f4.safetensors --remote
   wrangler r2 object put lightx2v-weights/v1/lightx2v-1.3b-f16.safetensors \
     --file web-weights/lightx2v-1.3b-f16.safetensors --remote
   wrangler r2 object put lightx2v-weights/v1/taew2_1.safetensors \
     --file web-weights/taew2_1.safetensors --remote
   wrangler r2 object put lightx2v-weights/v1/gallery/gallery.json \
     --file web-weights/gallery/gallery.json --remote
   # + one PUT per gallery embeds file under v1/gallery/
   ```
   Expected sizes: f16 2,979,733,280 B, TAEHV 22,642,902 B, f4
   877,080,928 B — all three now verified from the local export
   (see §10 Verified). Gallery embeds ~8 MB each (`prompt_embeds`
   [512, 4096] F32). Verify after upload with a Range GET against
   the public URL before pointing the page at it.
4. **Prompts** — no UMT5 in the browser (11 GB). Browser prompts come
   from the precomputed gallery (`web/gallery-prompts.txt`, embedded
   with `scripts/encode_prompt.py`, which needs torch + transformers
   + ~12 GB RAM — a user's-machine step). `serve` mode is the
   arbitrary-prompt path only once UMT5 lands in Phase 3; until then
   it encodes via a gallery embeds file / `--embeds`. The page says
   so plainly ("browser prompts come from the gallery — arbitrary
   prompts need the local server") and refuses non-gallery prompts
   in browser mode.

### Fastwan parity

| fastwan-rs | lightx2v-rs |
| --- | --- |
| `fastwan quantize --transformer … --weights int4-hqq-g64-ffn --out …` | `lightx2v quantize --transformer … --weights int4-hqq-g64[-ffn] --out …` (`export.rs` is a port of fastwan's, plus an Fp8 arm) |
| `fastwan-1.3b-f4/f16.safetensors`, `taew2_1.safetensors` | `lightx2v-1.3b-f4/f16.safetensors`, `taew2_1.safetensors` |
| `fastwan serve` | `lightx2v serve` (same shape + MemRoPE chunk progress) |
| `fastwan-web` (`createEngine`, `addBlock`, `finishDit`, `loadTae`, `generate`, `prepared`/`step`/`frames`, `try_download_async`) | `lightx2v-web` (same shape + `chunk` events, `download_async`; `DitBuilder` is the incremental loader) |
| `web/demo.html` (`PUBLIC_WEIGHTS`, `?weights=`, Cache API, `readGroups` guard) | `web/studio.html` (same shape) |
| `scripts/encode_prompt.py` | `scripts/encode_prompt.py` (same + `--prompt-file` gallery mode) |
| `scripts/deploy-pages.sh` → fastwan.pages.dev | `scripts/deploy-pages.sh [--weights …]` → lightx2v.pages.dev |
| 3-step DMD `[1000, 757, 522]`, shift 8 | **4-step DMD `[1000, 750, 500, 250]`, shift 5.0** — the deliberate divergence |
| x4 upscaler (Real-ESRGAN compact) | later stretch goal, not v1 |

### API contract

| Endpoint | Contract |
| --- | --- |
| `GET /`, `/index.html` | `web/studio.html` (include_str, no CDN deps) |
| `GET /api/health` (+ `/api/info` alias) | `{ready, error, backend, model, synthetic, gallery, max_pixels: 399360, max_frames: 1005, fps: 16, queued, load_s, loading}` |
| `GET /api/gallery` | `{prompts, restricted}` — the loaded gallery when one restricts prompts, else the bundled `web/gallery-prompts.txt` examples |
| `POST /api/generate` | `{prompt, width, height, frames, seed?}` → `{id, seed}`; frames must be 81 or 1005; real mode requires 832×480; JSON-only + Origin guard (403), engine not ready (503), queue full (429), unknown gallery prompt (400) |
| `GET /api/jobs/<id>` | `{id, prompt, stage, seed, width, height, total_frames, frames, chunks_done, total_chunks, tokens, encode_s, prepare_s, steps, total_steps, decode_s, elapsed_s, total_s, error}`; stages: queued → encoding → preparing → denoising → decoding → done/failed |
| `GET /api/jobs/<id>/frames?from=N` | binary: u32 LE width, height, first, count, then `count` frames of interleaved RGB8 |

Synthetic embeddings are seeded randoms from the prompt hash (FNV-1a
over the prompt bytes → `Rng::new(seed).normal(text_len·text_dim)` in
`serve`; the page's `tinyEmbeds` builds the same distribution in JS
for the tiny browser model), so the same prompt + seed gives the same
synthetic clip in either mode. The gallery format is fastwan's
(`gallery.json` + one safetensors file per prompt, tensor
`prompt_embeds [tokens, 4096]` F16/F32, only real tokens stored,
zero-padded to 512 by the engine) — see `scripts/encode_prompt.py`.
Note the two shapes: `encode_prompt.py` writes full `[512, 4096]`
files (rows past the real tokens zeroed, as Diffusers does), while the
serve gallery also accepts trimmed `[tokens, 4096]` files.

### Verified vs stubbed (2026-11-13, this VM)

- **Verified end to end, synthetic:** `lightx2v serve --synthetic`
  smoke test — health `ready: true` on `cpu`, gallery lists the 12
  bundled prompts, POST returns `{id: 1, seed: 7}`, the job reaches
  `done` with 9 frames / 1 chunk / 4 DMD steps, and
  `/frames?from=0` returns exactly 110,608 bytes (16 + 9·64·64·3).
  Debug build: 7.29 s total (6.99 s decode — debug CPU convs; release
  will be much faster). The non-JSON POST guard returns 403 and a
  missing job 404. The real `--model-dir` error path is verified too:
  an empty dir fails the engine at startup and `/api/health` reports
  `ready: false` with the weights error.
- **Verified, page vs live (2026-11-13, user's screenshot):** the live
  page showed two false errors at once. (1) "Cannot reach the weights":
  a public r2.dev URL serves NO CORS headers by default — the GET had
  no `Access-Control-Allow-Origin` and the Range preflight was 403 —
  fixed by `wrangler r2 bucket cors set` (GET + HEAD, `Range` header,
  ACAO `*`); after a few seconds' propagation the preflight returned
  204 and Range GETs 206. (2) "server not reachable at 127.0.0.1:7860":
  the page probed the local server on load even in browser mode — the
  init now only probes when server mode is selected, and browser mode
  sets a neutral engine line. The gallery is now checked BEFORE the
  ~900 MB download (in `showBrowserInfo`) and is optional at load
  (`bwLoad` warns instead of throwing after the download), so a
  missing `gallery.json` reads as "embeddings aren't uploaded yet",
  not as broken weights. Live verification of the fixed page is still
  owed (parent re-deploys + re-checks the live URL).
- **Verified, export (CPU, tiny):** `export::quantize_dit` on
  `WanConfig::tiny()` — stored codes equal load-time quantization
  per linear, and the written file re-reads block-ordered
  (`export_is_block_ordered`).
- **Verified, export (CPU, real weights):** the fp32
  `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` transformer (5,959,377,056 B,
  885 tensors, all F32) quantizes to `lightx2v-1.3b-f4.safetensors`
  (877,080,928 B — byte-identical in size to fastwan's f4 file,
  1545 tensors, 446.3 s) and `lightx2v-1.3b-f16.safetensors`
  (2,979,733,280 B — byte-identical in size to fastwan's f16 file,
  885 tensors, 108.3 s). Tensor-by-tensor parity vs the original
  (330 linears): f16 worst relative error 4.74e-04 (f16 rounding),
  f4 worst 6.55e-02 (int4 HQQ error). Both files are block-ordered
  (head, then blocks 0–29, contiguous) with
  `lightx2v.format=quant/1` metadata. The f16 export initially
  OOM-killed the VM (the writer built a second whole-file copy);
  `export::write` now streams to disk. The exported f4 file loads
  through the real `LightDit::load_with` path: `lightx2v generate
  --model-dir` (f4 + TAEHV, synthetic embeds) prints "loaded real
  DiT + TAEHV in 34.6 s".
- **Verified, wasm:** `crates/lightx2v-web` builds for
  `wasm32-unknown-unknown` in release and `scripts/build-web.sh`
  produces `web/pkg/` (`lightx2v_web.js` 54,658 B,
  `lightx2v_web_bg.wasm` 695,946 B); `lightx2v-wgpu` builds for
  wasm32 with the `webgpu` feature, and `cargo clippy --target
  wasm32-unknown-unknown -p lightx2v-web -p lightx2v-wgpu -- -D
  warnings` is clean (two wasm-only lints were real: an
  `arc_with_non_send_sync` allow with a single-threaded-wasm
  justification, and `await_holding_refcell_ref` fixed by cloning
  `Rc` model handles out before the first await). `cargo test
  --workspace` stays green on native (39 passed) because the web
  crate is an empty rlib off wasm32 (fastwan-web pattern). The
  generated JS was NOT run in a browser here: headless WebGPU cannot
  run on this VM (no GPU; WebGPU is a black canvas in headless
  Chromium), so browser execution is unverified until a real
  browser/phone loads the page with real weights.
- **In flight, not verified yet:** the gallery embeds cannot be
  produced on this VM (UMT5 needs ~12 GB RAM) — that is a
  user's-machine step via `scripts/encode_prompt.py`. Real generation
  on this VM is **blocked on RAM, with numbers**: the exported f4
  file loads through the real path ("loaded real DiT + TAEHV in
  34.6 s"), but the 81-frame denoise at 832×480 is OOM-killed
  (exit 137) — the DiT as f32 is 5.96 GB (1.49B params) and the
  MemRoPE KV cache at 832×480 is another ~6.6 GB (30 layers × 12
  cache frames × 1560 tokens × 1536 dim × K/V × 4 B), against
  7.9 GB total. A real generation needs a bigger machine or a GPU;
  the browser path does not have this problem (weights stay packed
  on GPU VRAM).
- **Stubbed / not built:** UMT5 (Phase 3 — `serve` without a gallery
  still errors naming the encoder; `generate` without `--embeds`
  does the same); the R2 upload (credentials + bucket are the
  parent's step — the exact commands are above; the files to upload
  already exist in `web-weights/`); the Cloudflare Pages deploy
  (no project/credentials in this task — `scripts/deploy-pages.sh`
  builds `web/dist/` and prints the deploy command, it does not run
  it); the x4 upscaler (stretch goal); a real CPU generation on this
  VM (RAM-blocked, see above).

### Phase 3: what the studio still needs

- **UMT5 encode:** the text encoder port unblocks real `serve`
  (any prompt) and native `encode` (gallery generation on the
  machine that has the RAM). Until then, prompts come from
  precomputed gallery embeddings.
- **Real-browser proof:** load the hosted page on a real browser/phone
  with real weights, run one gallery prompt at 480×272, and confirm
  `prepared` → steps → chunks → frames all land and play. Headless
  gates cannot judge this.
- **LTX / H3 variants:** out of scope for the studio — the studio
  serves the Wan 2.1 DMD student only; an LTX or H3 backbone would be
  a new `WanConfig`-shaped model behind the same `CausalPipeline`,
  not a studio change.
