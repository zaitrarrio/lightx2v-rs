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
crates/lightx2v-cli    `lightx2v` binary: plan / check / bench / generate
                       (--backend cpu|wgpu|auto, default auto)
scripts/reference/     make_fixture.py — tiny safetensors fixture generator
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
`--model-dir` with a Wan2.1-T2V-1.3B DMD checkpoint and errors cleanly
otherwise; real-checkpoint generation is Phase 3 (UMT5 text encoding
is still missing).

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
