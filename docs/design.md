# Design — lightx2v-rs

Phase 1 design notes. Everything in "Verified" was actually run on this
CPU-only VM on 2026-11-13; numbers quoted from the papers / upstream
configs are marked as such and were not re-measured here.

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
  Phase 2);
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
scales, the LightX2V headline quant) is Phase 2: the h3-rs contract
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
  --workspace --offline`: **34 passed, 0 failed** (28 core, 2 cpu,
  4 conformance). `cargo clippy --workspace --offline -- -D warnings`:
  clean. `cargo fmt --check`: clean.
- `lightx2v check`: **11/11 passed** (linear, layer_norm,
  attention, causal_attention_mask, sparse==dense@0-sparsity-gate-0,
  patchify/unpatchify roundtrip, dmd_step, quant q8/q4/fp8,
  rope origin identity, memrope cache bounded + sinks + ema,
  history-contrast residual).
- Conformance (`crates/lightx2v-model/tests/conformance.rs`):
  2-chunk tiny-DiT rollout vs an independent f32 reference
  (`tests/reference.rs`), max abs error **7.2e-7**; KV-cache test
  bounds the cache at 12 frames and keeps the sinks; TAEHV decode
  asserts `NotImplemented`; quantize flag parses. The test asserts
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
- `lightx2v generate --synthetic --out clip.latent`: 3 latent
  frames (9 video frames), 768 elems, **0.215 s**; writes raw f32
  latents (TAEHV decode is Phase 2). `generate` without
  `--synthetic` errors cleanly naming the missing model dir.

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

## 8. What is next (Phase 2/3)

- **Phase 2**: wgpu backend (+ CUDA if a GPU is available),
  NVFP4 micro-block quant, real DiT weights (`--model-dir`
  generation end-to-end), TAEHV decode (the `TaeHv::decode`
  interface exists; it returns `NotImplemented` now), UMT5 text
  encoding (embeddings are zeros in Phase 1), relativistic RoPE
  for open-ended streams.
- **Phase 3**: H3 backbone variant, Cloudflare edge serve
  (fastvideo-edge pattern), sequence-parallel / multi-GPU plan,
  `docs/runs/` measurements on named hardware. No speed claim
  until measured.
- Never deploy or quote the paper's VBench numbers as measured
  here — they are the upstream paper's, cited in §1.
