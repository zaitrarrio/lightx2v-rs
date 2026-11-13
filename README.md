# lightx2v-rs

A Rust port of **LightX2V** (ModelTC/LightX2V, Apache 2.0) with
**History-Forcing** (arXiv:2611.02782, Kuang et al.) — few-step autoregressive
video diffusion on the Wan 2.1 / Wan 2.2 family, in the fastwan-rs / h3-rs
shape: one fused `Backend` op vocabulary, a CPU f32 reference backend that
every other backend is checked against, and a causal chunk pipeline with a
bounded MemRoPE KV cache.

Phase 1 was a real, compiling, tested scaffold: the full core math
(config / schedule / RoPE / MemRoPE / history-contrast / quant /
safetensors loading / VRAM plan), a CPU reference backend, a DiT over the
`Backend` trait, a causal 3-latent-frame chunk pipeline with a 4-step DMD
sampler, and a `lightx2v` CLI. **Phase 2 (this version) adds the wgpu
backend and a real TAEHV decode**: the full `Backend` trait runs on
WebGPU (f16-packed weights, fused kernels, VSA sparse attention, the
MemRoPE KV cache on-device), `TaeHv::decode` decodes latents to RGB on
both backends, and the CLI gains `--backend cpu|wgpu|auto`. Real DiT
weights, UMT5 text encoding, and CUDA are Phase 3 — see
`docs/design.md`.

## What it implements (verified numbers)

History-Forcing (from `dmd/long_video/config.py` + `inference.yaml` in the
upstream clone, `research/history-forcing-upstream/`):

- 60 s demo: **252 latent frames** (1005 video frames), 832×480, 16 fps,
  bf16, batch 1, seed 0, CRF 23.
- MemRoPE **cache_frames 12 = sink 3 + long-EMA 1 + short-EMA 1 + recent 4
  + new 3**; `ema_alpha_long 0.01`, `ema_alpha_short 0.1`, `ema_adaptive
  false`, `use_block_rope true`, `num_frame_per_block 3`,
  `warp_denoising_step true`, `independent_first_frame false`,
  `context_noise 0`.
- DMD steps **[1000, 750, 500, 250]**, `timestep_shift 5.0`, 4 steps, no CFG.
- Paper results (not re-measured here): DMD student on Wan2.1-T2V-1.3B,
  60 s consistent, **VBench 85.07 @ 15 s, VBench-Long 79.82 @ 60 s**;
  OPS-D student on DreamX-World-5B (Wan2.2-TI2V-5B).

LightX2V (from the upstream clone, `research/lightx2v-upstream/`):

- 4-step distillation, no CFG, NVFP4/FP8 quant, sparse attention, block
  offload; up to 50× on RTX 5090 / H100 (upstream claim, not re-measured
  here). Wan 2.1/2.2, LTX-2, HunyuanVideo 1.5, MiniMax H3, Self-Forcing
  profiles. `wan_moe_t2v_distill.json`: `infer_steps 4`,
  `denoising_step_list [1000, 750, 500, 250]`, `sample_shift 5.0`,
  `enable_cfg false`, `boundary_step_index 2`, `cpu_offload true`,
  `offload_granularity model`, high/low-noise checkpoints.

## Layout

```
crates/lightx2v-core   config, schedule, rope, memrope, history-contrast,
                       quant (Q8/Q4/FP8 e4m3), safetensors loader, Backend
                       trait, VRAM plan
crates/lightx2v-cpu    CPU f32 reference backend (the oracle)
crates/lightx2v-wgpu   WebGPU backend: full Backend trait, WGSL shaders,
                       f16-packed weights, VSA sparse attention, MemRoPE
                       KV cache on-device
crates/lightx2v-model  DiT over Backend, causal chunk pipeline, 4-step DMD
                       sampler, TAEHV decode, conformance checks
crates/lightx2v-web    in-browser engine: the tiny synthetic model on the
                       CPU backend behind a wasm-bindgen API (wasm32 only)
crates/lightx2v-cli    `lightx2v` binary: plan / check / bench / generate /
                       serve (--backend cpu|wgpu|auto, default auto)
web/studio.html        Prompt Studio single-file UI (no CDN deps)
web/gallery-prompts.txt  12 prompts written for long consistent rollouts
scripts/build-web.sh   wasm build into web/pkg/ (scripts/deploy-pages.sh
scripts/build-site.py  builds web/dist/ for Cloudflare Pages; no weights)
scripts/encode_prompt.py  STUB: gallery format for precomputed UMT5 embeds
docs/design.md         what is fused and why, what is verified, what is next
```

## Backends

| Backend | Status | Notes |
| --- | --- | --- |
| `cpu` | Reference, verified | f32, straightforward loops; the oracle every other backend is checked against. `lightx2v check --backend cpu`: 11/11. |
| `wgpu` | Implemented, conformance **pending a real GPU** | Full `Backend` trait on wgpu 30 (Vulkan / WebGPU). Builds offline; on this VM no adapter answers (no Vulkan driver, lavapipe unavailable), so its 3 conformance tests exercise the clean skip path and its 11 checks have not run on hardware yet. Run `lightx2v check --backend wgpu` on a machine with a GPU. |
| `cuda` | Phase 3 | Not started. |

`--backend auto` (the default) uses wgpu when an adapter answers, else
the CPU. When the selected backend is unavailable the command prints a
clean skip reason and exits 0 — a missing GPU is not a failure. The
wgpu crate is behind the CLI's `wgpu` feature (on by default);
`cargo build -p lightx2v-cli --no-default-features` builds CPU-only.

## Build / test / run

```sh
cargo build --workspace --offline
cargo test --workspace --offline        # 37 passed, 0 failed (this VM;
                                        # 3 wgpu tests skip: no adapter)
cargo clippy --workspace -- -D warnings # clean
cargo fmt --check                       # clean

./target/debug/lightx2v plan            # 81-frame clip, 16 GiB budget
./target/debug/lightx2v plan --frames 1005 --offload block
./target/debug/lightx2v check                       # 11 checks, 11 passed (cpu)
./target/debug/lightx2v check --backend wgpu       # 11/11, or a clean skip
./target/debug/lightx2v bench --layers 1
./target/debug/lightx2v generate --synthetic --out clip
# -> clip.latent (raw f32) + clip.rgb (interleaved RGB8, 9 frames 64x64)
```

`plan` for the 81-frame clip at 16 GiB: peak **7638.8 MiB, fits**. The
60 s demo (1005 frames) peaks **17992.6 MiB** with no offload — it does
not fit 16 GiB; `--offload block` keeps 189,199,572 of 2,837,993,600
weight bytes resident. `generate` without `--synthetic` needs
`--model-dir` with a Wan2.1-T2V-1.3B DMD checkpoint plus `--embeds` with
precomputed UMT5 prompt embeddings (see `scripts/encode_prompt.py`); it
errors cleanly otherwise. The UMT5 text encoder itself is Phase 3.

## Prompt Studio

`lightx2v serve` is the lightx2v equivalent of fastwan's Prompt Studio —
a local engine plus a single-file web UI (`web/studio.html`), not a
hosted API. The engine thread owns the backend and the resident DiT/TAEHV
and runs jobs one at a time from a queue (max 3 queued); the HTTP thread
only reads the shared job table, so polling never waits on the engine.

```sh
./target/debug/lightx2v serve --synthetic
# -> http://127.0.0.1:7860/  (studio page, health, gallery, jobs)
./target/debug/lightx2v serve --synthetic --allow-origin https://lightx2v.pages.dev
./target/debug/lightx2v serve --model-dir <ckpt> --gallery gallery/
./target/debug/lightx2v quantize --transformer <ckpt>/transformer --weights int4-hqq-g64 --out web-weights/lightx2v-1.3b-f4.safetensors
./target/debug/lightx2v generate --model-dir <ckpt> --embeds fox.safetensors --frames 81 --out clip
```

| Mode | Prompts | Output |
| --- | --- | --- |
| `--synthetic` | any prompt; embeddings are seeded randoms from the prompt hash | tiny random model: 64×64, 9 frames, 1 chunk (structured noise — it proves the path, not the picture) |
| `--synthetic --gallery <dir>` | only that gallery's prompts | same toy output, gallery-restricted |
| `--model-dir <dir> --gallery <dir>` | only that gallery's prompts (precomputed UMT5 embeddings) | real 1.3B DiT, 832×480, 81 or 1005 frames, 7 or 84 MemRoPE chunks |
| `--model-dir <dir>` (no gallery) | — | engine fails to start and names the missing UMT5 encoder |

API: `GET /` (studio page), `GET /api/health` (ready/backend/model/
synthetic/gallery/max), `GET /api/gallery`, `POST /api/generate`
`{prompt, width, height, frames, seed?}` → `{id, seed}` (JSON-only +
Origin guard, 429 past 3 queued), `GET /api/jobs/<id>` (stage, per-step
and per-chunk timings, frames decoded so far, `chunks_done` /
`total_chunks` for the long-video rollout), `GET /api/jobs/<id>/frames?from=N`
(binary: u32 LE width, height, first, count, then RGB8). Frames decode
progressively via `TaeHv::decode_chunk`, so they land as they decode.
An engine panic surfaces into the job error and `/api/health`.

**In this browser:** the page's browser mode loads `web/pkg/` (built by
`scripts/build-web.sh` from `crates/lightx2v-web`) and runs the real
Wan2.1-T2V-1.3B DMD student on WebGPU — weights stream from a public R2
bucket, one transformer block per HTTP Range request, into the browser
cache (downloaded once), via `createEngine("lightx2v")` +
`addBlock`/`finishDit`/`loadTae`/`generate` with `prepared`/`step`/`chunk`/
`frames` events. Two weight files: **f4** = int4-hqq-g64 block linears,
everything else f16 (~0.88 GB) and **f16** = everything f16 (~2.98 GB),
plus TAEHV (~23 MB), produced by `lightx2v quantize` (see below) and
written block-ordered — the page **refuses** files that are not
block-ordered. With no weights configured the page offers the tiny
synthetic model (`createEngine("tiny")`, `?model=tiny`) as the dev
fallback. Browser prompts come from the gallery — there is no text
encoder in the browser (UMT5-XXL is 11 GB); arbitrary prompts need the
local server. `scripts/deploy-pages.sh [--weights <base URL>]` builds
`web/dist/` (studio page + wasm pkg + gallery prompts + `manifest.json`,
never weights) for Cloudflare Pages; it does not deploy by itself.
`?weights=<base URL>` points the page at other weights.

**Weights export:** `lightx2v quantize --transformer <dir> --weights
<int4-hqq-g64 | native> --out <file>` reads a HF
FastWan2.1-T2V-1.3B-Diffusers transformer dir and writes a block-ordered
safetensors file under `web-weights/` (git-ignored, never committed).
`scripts/encode_prompt.py` produces the gallery embeds (needs
torch + transformers + ~12 GB RAM — a user's-machine step, not this VM).

**Fastwan parity:** the studio mirrors fastwan-rs piece for piece:

| fastwan-rs | lightx2v-rs |
| --- | --- |
| `fastwan quantize --transformer … --weights int4-hqq-g64-ffn --out …` | `lightx2v quantize --transformer … --weights int4-hqq-g64[-ffn] --out …` |
| `fastwan-1.3b-f4.safetensors` (0.88 GB) / `fastwan-1.3b-f16.safetensors` (2.98 GB) / `taew2_1.safetensors` (23 MB) | `lightx2v-1.3b-f4.safetensors` / `lightx2v-1.3b-f16.safetensors` / `taew2_1.safetensors` (same sizes, same sources) |
| `fastwan serve` (DiT + TAEHV resident, `--synthetic`, `--allow-origin`, JSON-only + Origin guard) | `lightx2v serve` (same shape + MemRoPE chunk progress) |
| `fastwan-web` (`createEngine`, `addBlock`, `finishDit`, `loadTae`, `generate`, `prepared`/`step`/`frames`) | `lightx2v-web` (same shape + `chunk` events for the causal rollout, `download_async` readback) |
| `web/demo.html` (`PUBLIC_WEIGHTS`, `?weights=`, Cache API, `readGroups` block-order guard) | `web/studio.html` (same shape, `lightx2v-weights-v1` cache) |
| `scripts/encode_prompt.py` (UMT5 → `prompt_embeds` gallery) | `scripts/encode_prompt.py` (same, plus `--prompt-file` gallery mode) |
| `scripts/deploy-pages.sh` → fastwan.pages.dev | `scripts/deploy-pages.sh [--weights …]` → lightx2v.pages.dev |
| 3-step DMD schedule `[1000, 757, 522]`, shift 8 | **4-step DMD schedule `[1000, 750, 500, 250]`, shift 5.0** — the one deliberate divergence: the schedule is a property of the History-Forcing student, not the weights |
| x4 upscaler (Real-ESRGAN compact) | later stretch goal, not v1 |

**Synthetic vs real, plainly:** everything end-to-end is verified in
synthetic mode (serve smoke test: job reaches `done`, 9 frames,
110,608 frame bytes; wasm builds and `build-web.sh` produces
`web/pkg/`). Real weights are exported and verified on CPU (see below)
— but a real generation on this VM and any browser execution are **not
yet verified**: the gallery embeds need a 12 GB machine, and headless
WebGPU cannot run here, so the only real gate for browser execution is
a real browser/phone. See `docs/design.md` §10 for the exact status.

## Lineage

- **fastwan-rs** — the template: workspace layout, fused `Backend`
  trait, CPU reference as ground truth, mmap safetensors loader,
  conformance-test discipline. Code is reimplemented with lightx2v
  names, not copied.
- **h3-rs** — MiniMax H3, one of LightX2V's supported backbones
  (`configs/minimax_h3*`, `streamrender_h3`, `taomate_h3` upstream).
  Contributes the variant-preset pattern (mirrored as `DistillConfig`
  profiles), the AdaLN-streaming idea (modulation tables computed per
  generation, never resident — mirrored as `step_mods`), the NVFP4
  contract warning, and the check discipline (90/90 on a named GPU).
  h3-rs has **no** causal/autoregressive KV-cache chunked rollout —
  it denoises full clips, continuation via Ref2VA keyframes — so
  lightx2v-rs does **not** copy its pipeline; the causal chunk
  pipeline here follows History-Forcing + fastvideo-rs causal
  discipline instead. H3 as a backbone is a Phase 3 variant.
- **fastvideo-rs** — the upstream fastwan-rs was a lean
  re-implementation of. Inherited: the mmap safetensors loader
  (fastvideo-loader → `weights.rs`), FP8 E4M3 ops (fastvideo-ops →
  `quant.rs`), golden/parity test discipline (gold-check/gold-run →
  conformance tests + `docs/runs`-style notes), and the Cloudflare
  edge Worker + Durable Object idea (fastvideo-edge → Phase 3 serve).
  Deliberate divergence: fastwan-rs / lightx2v-rs scope to one model
  family + consumer GPUs + a fused `Backend` trait; fastvideo-rs is
  datacenter CUDA-first (cudarc primary, Runpod RTX PRO 6000,
  sequence parallel, full serve/autoscale/deploy stack). The causal
  discipline is taken from fastvideo-rs `wan/causal.rs`: a block's
  K/V is **appended when new, overwritten in place when the same
  block is re-run** on the next denoising step — implemented here as
  `Backend::kv_cache_put`. lightx2v-rs differs in that it attends
  through an explicit causal block mask (`tk <= tq`) plus MemRoPE
  EMA compression and denoises each chunk fully once before moving
  on (History-Forcing inference, `warp_denoising_step`); fastvideo-rs
  causal attends maskless to the last `max_attention` window and
  overwrites cache slots on re-run. Relativistic RoPE (un-roped cache,
  re-rope the window per forward) is a Phase 2/3 candidate for
  open-ended streams; lightx2v-rs currently stores roped K and
  rotates by original temporal indices (block RoPE).

## Rules

- CPU backend is ground truth; GPU code is trusted only after `check`
  passes on that hardware. No speed claim without a measurement on
  named hardware under `docs/runs/`.
- A missing weight key or a shape mismatch is an error, never a default.
- `research/` (upstream Python clones) is excluded from the push set
  by `.gitignore` and is not a workspace member.
