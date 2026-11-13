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
- **Features off by default.** No wgpu/cuda deps in Phase 1; GPU code
  is gated behind features that are OFF by default.

## Layout

- `crates/lightx2v-core` — pure math + traits, no backend: `config`
  (Wan/Tae/Generation/MemRope/Distill configs with the exact
  validation rules from history-forcing `config.py`), `geometry`,
  `rng`, `schedule` (DMD [1000,750,500,250], shift 5.0), `rope`,
  `memrope`, `history`, `quant` (Q8 group-128, Q4 HQQ, FP8 e4m3),
  `sparse` (VSA tile plan), `weights` (mmap safetensors), `backend`,
  `plan` (VRAM estimator).
- `crates/lightx2v-cpu` — `CpuBackend`, the f32 oracle.
- `crates/lightx2v-model` — `LightDit` (DiT over `Backend`),
  `CausalPipeline` (3-latent-frame chunks, 4 DMD steps per chunk,
  caches persist across chunks), `TaeHv` (load validates the manifest;
  `decode` is `NotImplemented` until Phase 2), `check.rs` (11 checks),
  `tests/conformance.rs` (tiny safetensors fixture vs an independent
  f32 reference rollout in `tests/reference.rs`).
- `crates/lightx2v-cli` — `lightx2v plan/check/bench/generate`.
- `scripts/reference/make_fixture.py` — pure-stdlib safetensors
  writer for the tiny DiT + tiny TAEHV fixtures.

## Commands

```sh
cargo build --workspace --offline
cargo test --workspace --offline
cargo clippy --workspace --offline -- -D warnings
cargo fmt --check
./target/debug/lightx2v check
./target/debug/lightx2v plan [--frames 1005] [--offload block]
./target/debug/lightx2v generate --synthetic --out clip.latent
```

## Phase 1 status (2026-11-13)

- `cargo test --workspace`: 34 passed, 0 failed (28 core, 2 cpu,
  4 conformance). `lightx2v check`: 11/11. Clippy `-D warnings`
  clean, `cargo fmt --check` clean.
- Conformance: 2-chunk tiny-DiT rollout vs the direct f32 reference,
  max abs error **7.2e-7** (f32 accumulation-order noise; the test
  asserts max-abs < 1e-5, not relative error, because relative error
  blows up on near-zero elements).
- Known stubs for Phase 2/3: TAEHV decode (`NotImplemented`), real
  DiT weights (generate needs `--model-dir`), UMT5 text encoding
  (embeddings are zeros in Phase 1), wgpu/CUDA backends, NVFP4
  micro-block quant, relativistic RoPE for open-ended streams,
  H3 backbone variant.
- Do not push: report the ghapi push-local command to the parent.
  `research/` is excluded from the push set.
