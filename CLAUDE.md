# CLAUDE.md — lightx2v-rs

Rust port of LightX2V + History-Forcing (Wan 2.1 / Wan 2.2 family), in the
fastwan-rs / h3-rs shape. Lineage is in `README.md` and `docs/design.md`;
code is reimplemented with lightx2v names, not copied from fastwan-rs,
h3-rs, or fastvideo-rs.

## Invariants

- **One op vocabulary.** `lightx2v_core::Backend` is the fused minimal
  trait (linear / layer_norm / residual_norm / gate_residual / head_split /
  attention / causal_attention / sparse_attention / patchify / unpatchify /
  dmd_step / kv-cache ops / MemRoPE update / weight_fp8 / offload plan).
  A new op = CPU impl + `check.rs` case + conformance coverage. No op
  lives only in one backend.
- **CPU is ground truth.** `lightx2v-cpu` is f32, straightforward loops,
  no unsafe shortcuts. GPU backends (Phase 2/3) are trusted only after
  `lightx2v check` passes on that hardware. No speed claim without a
  measurement on named hardware under `docs/runs/`.
- **Missing weight key or shape mismatch is an error**, never a default
  (`weights.rs`: `SafetensorsDir::get` returns `Error::Weights`).
- **Checkpoint contracts** (h3-rs lesson, Wan flavour): channel-major
  patch features (`patchify` feature `cc·p² + kh·p + kw`); `value, gate`
  halves of every SwiGLU `fc_in`; per-head `norm_q`/`norm_k`; RoPE over
  the first 96 of 128 head channels in h3 — Wan uses its own 3-axis
  RoPE (`rope.rs`, block RoPE for MemRoPE); AdaLN slot order
  shift/scale/gate (attn) then (mlp); `x0 = x + σ·v`.
- **Causal discipline** (fastvideo-rs `wan/causal.rs`): a chunk's K/V is
  appended when the block is new and **overwritten in place**
  (`Backend::kv_cache_put`) when the same block is re-run on the next
  DMD step. Never append duplicates — a 4-step chunk would occupy the
  cache four times over and evict real history. The overwrite search
  must skip the EMA slots: they share the first non-sink frame's
  temporal index.
- **v is never RMS-normalised.** `head_split` applies RMS only when a
  weight is given (q/k); with `rms_w = None` the slice passes through
  raw. (Found 2026-11-13: the CPU impl RMS'd v, diverging from
  Diffusers by 0.63 absolute.)
- **MemRoPE**: cache 12 = sink 3 + long-EMA 1 + short-EMA 1 + recent 4
  + new 3; alphas 0.01 / 0.1; EMA slots are seeded as zeros after the
  sinks on the first non-sink append; `MemRopeCache` update runs
  per-attention-layer after each layer; eviction folds the oldest
  recent frame into the EMA slots before dropping it.
- **History-contrast**: `corrected = base + w·(teacher_with − teacher_without)`.
- **Features off by default — except wgpu, which is on by default in
  the CLI.** The `wgpu` crate is behind the CLI's `wgpu` feature
  (`default = ["wgpu"]`); `cargo build -p lightx2v-cli
  --no-default-features` builds CPU-only, and `--backend wgpu` on
  that build prints a clean skip. No CUDA dep yet (Phase 3).

## Layout

- `crates/lightx2v-core` — pure math + traits, no backend: `config`
  (Wan/Tae/Generation/MemRope/Distill configs with the exact
  validation rules from history-forcing `config.py`), `geometry`,
  `rng`, `schedule` (DMD [1000,750,500,250], shift 5.0), `rope`,
  `memrope`, `history`, `quant` (Q8 group-128, Q4 HQQ, FP8 e4m3),
  `sparse` (VSA tile plan), `weights` (mmap safetensors), `backend`,
  `plan` (VRAM estimator).
- `crates/lightx2v-cpu` — `CpuBackend`, the f32 oracle.
- `crates/lightx2v-wgpu` — `WgpuBackend`, the full `Backend` trait on
  wgpu 30: f16-packed weights (`u32` pairs via `unpack2x16float`, no
  `shader-f16`), Q8/Q4 in-shader dequant, one flash kernel with dense /
  VSA-fine / causal-masked modes, VSA pool + coarse + top-k, the
  MemRoPE KV cache on-device (`[rows, heads·d]`, temporal/kinds on
  the host). Shaders in `src/shaders/`. `try_new()` returns `None`
  when no adapter answers — callers skip, never fail. `is_software()`
  reports a CPU rasterizer (fine for conformance, meaningless for
  timing).
- `crates/lightx2v-model` — `LightDit` (DiT over `Backend`),
  `CausalPipeline` (3-latent-frame chunks, 4 DMD steps per chunk,
  caches persist across chunks), `TaeHv` (load validates the manifest;
  `decode` decodes 4-latent-frame chunks to RGB on any backend;
  stored-quantized conv weights are dequantized on the host at load),
  `check.rs` (11 checks, `run_all_on<B: Backend>` for any backend),
  `tests/conformance.rs` (tiny safetensors fixture vs an independent
  f32 reference rollout in `tests/reference.rs`; TAEHV decode shape /
  determinism / chunk-invariance test).
- `crates/lightx2v-cli` — `lightx2v plan/check/bench/generate`,
  `--backend cpu|wgpu|auto` (default `auto`; unavailable backend =
  clean skip, exit 0).
- `scripts/reference/make_fixture.py` — pure-stdlib safetensors
  writer for the tiny DiT + tiny TAEHV fixtures.

## Commands

```sh
cargo build --workspace --offline
cargo test --workspace --offline
cargo clippy --workspace --offline -- -D warnings
cargo fmt --check
./target/debug/lightx2v check [--backend cpu|wgpu|auto]
./target/debug/lightx2v plan [--frames 1005] [--offload block]
./target/debug/lightx2v generate --synthetic --out clip
# -> clip.latent (raw f32) + clip.rgb (interleaved RGB8)
cargo build -p lightx2v-cli --no-default-features --offline  # CPU-only
```

## Phase 2 status (2026-11-13)

- `cargo test --workspace`: **37 passed, 0 failed** (28 core, 2 model,
  4 conformance, 3 wgpu — the wgpu 3 skip on this VM: no adapter).
  `lightx2v check --backend cpu`: 11/11. `check --backend wgpu`:
  clean skip, exit 0. Clippy `-D warnings` clean, `cargo fmt --check`
  clean.
- Conformance: 2-chunk tiny-DiT rollout vs the direct f32 reference,
  max abs error **7.2e-7** (f32 accumulation-order noise; the test
  asserts max-abs < 1e-5, not relative error, because relative error
  blows up on near-zero elements).
- TAEHV decode works on CPU: 3 latent frames → 9 video frames at
  64×64, deterministic on repeat decode, bit-identical whether
  decoded as one 6-frame chunk or 4+2 chunks (MemBlock memory carried
  across the boundary). `generate --synthetic` writes `clip.latent`
  + `clip.rgb` (9 frames, 64×64, 110,592 bytes).
- wgpu conformance is **not yet run on hardware**: no Vulkan driver
  on this VM (`/usr/share/vulkan/icd.d/` absent, lavapipe not
  installable — apt lists unreachable). The wgpu crate builds offline
  (wgpu 30, default-features off + `vulkan`/`wgsl` only; `pollster`
  0.4.0, the cached version), the shaders were reviewed against the
  CPU oracle line by line, and the tests assert the skip path — but
  no wgpu kernel has executed. First run on a real GPU:
  `lightx2v check --backend wgpu` (expect 11/11) and
  `cargo test -p lightx2v-wgpu` (expect the DiT rollout < 1e-5 and
  the decode < 1e-3 with f32 weights; the f16 decode error is
  printed and bounded at 5e-2).
- Known stubs for Phase 3: real DiT weights (`generate` needs
  `--model-dir` and still errors — UMT5 text encoding is the
  blocker), CUDA backend, NVFP4 micro-block quant, relativistic RoPE
  for open-ended streams, H3 backbone variant.
- Do not push: report the ghapi push-local command to the parent.
  `research/` is excluded from the push set.
