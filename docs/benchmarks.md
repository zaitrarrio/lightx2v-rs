# lightx2v-rs benchmarks

Measured 2026-11-13 on this VM (the profiling session). All commands are
exact; re-run them to reproduce. **Nothing in this file was measured on a
GPU** — see [What is NOT measured](#what-is-not-measured).

## Hardware

| | |
|---|---|
| VM | Linux x86_64, 4 vCPUs visible / 1.5 shared CPUs, 7,934 MiB RAM |
| GPU | **none** — no Vulkan driver, no lavapipe; `--backend wgpu` cleanly skips |
| Toolchain | rustc/cargo 1.99.1, release profile (`cargo build --release --offline`) |
| `perf` / flamegraph | **not available** (no `perf` binary); profiling is the in-tree
  `LIGHTX2V_PROFILE=1` per-op timer (see [Profiler](#profiler)) |
| RSS measurement | `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` after the child exits
  (no `/usr/bin/time` on this VM) |

Releases are what is timed throughout: the debug build is ~100x slower
on TAEHV decode (6.99 s vs 0.066 s for the toy clip) and is not
representative of anything.

## Profiler

`LIGHTX2V_PROFILE=1` enables per-op timing in the CPU backend
(`crates/lightx2v-cpu/src/lib.rs`): a RAII timer on every `Backend` op,
reported by `Backend::profile_report()` and printed by `bench`, `check`
and `generate`. Times are **inclusive** — `residual_norm` calls
`layer_norm`, so the `layer_norm` row is counted inside the
`residual_norm` row too. Overhead off is a single `OnceLock` load per
op call. The wgpu backend does not implement `profile_report()` yet, so
GPU runs only get wall-clock timings (see [GPU checklist](#gpu-checklist)).

```sh
LIGHTX2V_PROFILE=1 ./target/release/lightx2v bench --backend cpu --layers 30
LIGHTX2V_PROFILE=1 ./target/release/lightx2v generate --synthetic --out clip
```

## `bench` (tiny DiT, 1 chunk x 4 DMD steps)

Tiny geometry: 2 layers, dim 24, 48 tokens/chunk, 3 latent frames
(`WanConfig::tiny()`). `bench --layers N` scales the layer count only.

| run | layers | total | load | prompt | denoise |
|---|---|---|---|---|---|
| 1 | 2 | 0.005 s | 0.000 s | 0.000 s | 0.005 s |
| 2 | 2 | 0.005 s | 0.000 s | 0.000 s | 0.004 s |
| 3 | 2 | 0.007 s | 0.001 s | 0.000 s | 0.006 s |
| 1 | 30 | 0.066 s | 0.002 s | 0.001 s | 0.066 s |
| 2 | 30 | 0.064 s | 0.002 s | 0.000 s | 0.064 s |
| 3 | 30 | 0.065 s | 0.002 s | 0.001 s | 0.064 s |

Denoise scales ~13x from 2 to 30 layers (15x the layers; the rest is
fixed per-chunk overhead). The 30-layer runs are stable to ±1 ms; the
2-layer runs are noise-dominated (see the `attention` spike in run 3
below — a shared-CPU artifact, not a code path).

### Per-op profile, `bench --layers 30` (run 1; runs 2–3 agree to ±1%)

| op | calls | total | mean/call | share |
|---|---|---|---|---|
| `linear` | 760 | 44.08 ms | 58.0 µs | 65.2% |
| `causal_attention` | 120 | 12.88 ms | 107.3 µs | 19.0% |
| `attention` (cross) | 120 | 2.85 ms | 23.8 µs | 4.2% |
| `head_split` | 540 | 2.45 ms | 4.5 µs | 3.6% |
| `layer_norm` | 364 | 1.93 ms | 5.3 µs | 2.9% |
| `residual_norm` | 240 | 1.72 ms | 7.2 µs | 2.5% |
| everything else | — | <1.5 ms | — | <2% |

At toy scale the DiT is a `linear` benchmark: 65% of denoise time in
one scalar f32 GEMM loop. `causal_attention` is second at 19% — but
note the toy cache is tiny (12 frames x 4 tokens = 48 rows, so
`sk ≈ sq`); at real scale the cache makes `sk ≈ 5x sq` and the
attention share grows sharply (see [Real-scale model](#real-scale-model)).

`bench --layers 2` profile (run 1): `linear` 3.14 ms / 65.1%,
`causal_attention` 0.84 ms / 17.5%, `attention` 0.18 ms / 3.7%,
`layer_norm` 0.16 ms / 3.3%, `head_split` 0.15 ms / 3.2%. Run 3 showed
`attention` at 1.97 ms / 30.0% (mean 247 µs/call vs 22 µs in runs 1–2)
— a one-run spike on a shared CPU, not reproducible in the 30-layer
runs. Treat sub-millisecond op totals as noise.

## `check` (conformance, cpu)

```sh
./target/release/lightx2v check --backend cpu
# 11 checks, 11 passed, 0 failed; wall <20 ms (3–17 ms across runs)
```

All 11 checks pass (worst error: quant roundtrip 6.55e-2 vs tol 2e-1;
attention 5.96e-8 vs tol 1e-6). The check shapes are hand-sized (2–64
elements), so a per-op breakdown is meaningless here — the whole suite
is under 20 ms and any single op is timer noise. The per-op breakdown
that matters is the `bench`/`generate` profile above.

## `generate --synthetic` (toy clip, denoise + decode)

3 latent frames → 9 video frames, 64x64, 110,592 bytes RGB. Release,
3 runs:

| run | denoise | decode | wall | peak RSS |
|---|---|---|---|---|
| 1 | 0.005 s | 0.067 s | 0.08 s | 11.2 MiB |
| 2 | 0.005 s | 0.065 s | 0.08 s | 11.2 MiB |
| 3 | 0.005 s | 0.066 s | 0.08 s | 11.2 MiB |

**Decode is 93% of the toy clip**: 65–67 ms of a ~71 ms total.
Per-op profile (run 1): `conv2d` 65.3 ms / 92.0% (35 calls, mean
1.87 ms), `linear` 3.1 ms / 4.4%, `causal_attention` 0.8 ms / 1.2%.
Decode cost: ~22 ms per latent frame, ~7.3 ms per video frame at toy
scale. The decode path is the same code the server uses progressively
(see [Serve](#serve)).

## Serve (`lightx2v serve --synthetic`, release)

```sh
./target/release/lightx2v serve --synthetic --addr 127.0.0.1:7861
```

| measurement | value |
|---|---|
| `/api/health` ready | load 1.4 ms, `ready: true`, backend `cpu` |
| RSS at ready | 6 MiB |
| Queue 3 jobs (POST `/api/generate`) | 0.002 s for all three |
| RSS with 3 queued | 6 MiB |
| Job 1 / 2 / 3 `total_s` | 0.076 / 0.074 / 0.067 s |
| Job `decode_s` | 0.070 / 0.067 / 0.063 s (89–93% of job time) |
| Job DMD `steps` | 4 x ~1–2 ms each |
| Job `frames` / `chunks` | 9 frames, 1/1 chunk |
| Wall for 3 jobs (serial, incl. 0.2 s poll granularity) | 0.61 s |
| RSS after 3 jobs done | 10 MiB |
| `GET /api/jobs/1/frames?from=0` | 110,608 bytes; header `w=64 h=64 first=0 count=9` |

Decode-in-serve (`decode_chunk`, progressive) costs the same as
whole-clip decode in the CLI (62–70 ms vs 65–67 ms): `TaeHv::decode`
loops the same `launch_chunk` internally, so the chunked path adds no
overhead at this scale. Job-table memory at toy scale is ~0.3 MiB per
finished job (110,592 B of frames); with `KEEP_JOBS 8` that is ~0.9 MiB.
**At real scale each finished job holds 81 x 832 x 480 x 3 = 97 MiB of
frames, so 8 kept jobs is ~777 MiB** — the job table, not the engine,
is the serve path's memory risk at 832x480 (see bottleneck 4).

## `plan` (memory model, 16 GiB budget, f16 weights)

| frames | offload | weights resident | MemRoPE cache | DiT activations | decode peak | peak | fits |
|---|---|---|---|---|---|---|---|
| 81 | none | 2,706.5 MiB | 3,290.6 MiB | 1,551.6 MiB | 1,657.5 MiB | 7,638.8 MiB | yes |
| 1005 | none | 2,706.5 MiB | 3,290.6 MiB | 11,905.5 MiB | 1,657.5 MiB | 17,992.6 MiB | **no** |
| 1005 | model | 2,706.5 MiB | 3,290.6 MiB | 11,905.5 MiB | 1,657.5 MiB | 17,992.6 MiB | **no** |
| 1005 | block | 180.4 MiB | 3,290.6 MiB | 11,905.5 MiB | 1,657.5 MiB | 16,302.2 MiB | yes |
| 1005 | disk | 180.4 MiB | 3,290.6 MiB | 11,905.5 MiB | 1,657.5 MiB | 16,302.2 MiB | yes |
| 1005 | none, int8 weights | 1,353.3 MiB | 3,290.6 MiB | 11,905.5 MiB | 1,657.5 MiB | 16,639.4 MiB | **no** |
| 81 | none, f32 weights | 5,413.0 MiB | 3,290.6 MiB | 1,551.6 MiB | 1,657.5 MiB | 10,345.3 MiB | yes |

Notes: the 60 s rollout's peak is dominated by DiT activations
(11,905.5 MiB — the plan notes the FFN runs in 1,536 chunks of 256
tokens); block/disk offload is what makes 60 s fit in 16 GiB, and
model-granularity offload does not change the peak at all. The MemRoPE
cache is bounded at 12 frames in every row (7 chunks of 3 at 81 frames,
84 at 1005).

## Quantize / export (real weights — from the Studio build, not re-run)

The fp32 `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` transformer
(5,959,377,056 B, 885 tensors, all F32) was exported on this VM during
the Prompt Studio build; re-running a 6 GB export for this benchmark
would add nothing, so the recorded numbers stand:

| export | size | tensors | time | parity vs original (330 linears) |
|---|---|---|---|---|
| `int4-hqq-g64` (f4) | 877,080,928 B | 1,545 | 446.3 s | worst rel err 6.55e-02 |
| `native` (f16) | 2,979,733,280 B | 885 | 108.3 s | worst rel err 4.74e-04 |

Both files are block-ordered (head, then blocks 0–29, contiguous).
The exported f4 loads through the real `LightDit::load_with` path in
34.6 s (`lightx2v generate --model-dir`). The f16 export initially
OOM-killed the VM (the writer built a second whole-file copy);
`export::write` now streams to disk.

## wasm / in-browser cost model (no execution numbers)

`web/pkg/` as built by `scripts/build-web.sh` (release wasm):

| file | bytes |
|---|---|
| `lightx2v_web.js` | 54,648 |
| `lightx2v_web_bg.wasm` | 695,933 |
| `web/dist/index.html` (studio page) | 34,572 |

Weights streamed from R2 (`pub-…r2.dev/v1`, manifest 562 B):

| file | bytes |
|---|---|
| `lightx2v-1.3b-f4.safetensors` | 877,080,928 |
| `lightx2v-1.3b-f16.safetensors` | 2,979,733,280 |
| `taew2_1.safetensors` | 22,642,902 |

The page fetches `manifest.json`, reads the safetensors header with
two Range requests, groups tensors into head + `blocks.0–29`, refuses
the file if any group is not contiguous, then fetches one Range request
per group (30 blocks + head + TAEHV ≈ 32 requests), cached in the
Cache API (`lightx2v-weights-v1`, keyed by file + size + group).
Browser execution is **unverified here**: headless WebGPU cannot run
on this VM (no GPU; WebGPU is a black canvas in headless Chromium).
One deployment gotcha already found and fixed on the live page:
public `r2.dev` serves **no** `Access-Control-Allow-Origin` by
default, so the bucket needs an explicit CORS rule (`wrangler r2
bucket cors set`) before the page's cross-origin Range fetches work
(preflight now 204, Range 206).

## Real-scale model (analytic — derived, not measured)

Per-layer-step FLOPs at production geometry (dim 1536, 12 heads x 128,
FFN 8960, chunk 4,680 tokens = 3 latent frames x 1,560 tokens/frame,
text 512 x 4096, steady-state cache 18,720 rows = 12 frames):

| op (per layer-step) | FLOPs | share |
|---|---|---|
| linears (qkv + out + cross q/out + FFN) | 3.91e11 | 38% |
| causal self-attention (QK + PV, ~90% of pairs allowed by the mask) | 6.05e11 | 60% |
| cross-attention | 1.47e10 | 1% |
| norms / head_split / sampler | ~1e9 | <1% |
| **per layer-step** | **~1.01e12** | |
| per chunk (30 layers x 4 steps) | 1.22e14 | |
| per 81-frame clip (7 chunks) | 8.6e14 | |
| TAEHV decode, whole 81-frame clip | 6.9e12 | 0.8% of denoise |

Two inversions vs the toy measurements: **attention, not linears, is
the dominant DiT cost at real scale** (60% vs 19% measured at toy
scale, because the MemRoPE cache makes `sk ≈ 5x sq`), and **decode,
93% of the toy clip, is under 1% at real scale**. Any optimization
prioritized from toy numbers alone would aim at the wrong op.

Measured per-op rates on this CPU (extrapolation basis, pessimistic —
tiny matrices understate the rate; real matrices are more
cache-friendly per FLOP): `linear` ≈ 1.4 GFLOPS (61.9 MFLOP in
44.1 ms at 30 layers), `conv2d` ≈ 0.4 GFLOPS (23.7 MFLOP estimated
from the TAE config over the toy decode, 64 ms measured). Projected
CPU time for one 81-frame clip: denoise ≈ 8.6e14 / 1.4e9 ≈ 170 h,
decode ≈ 6.9e12 / 0.4e9 ≈ 5 h. The CPU backend is a conformance
oracle at real scale, not an inference engine — which is why the
real numbers must come from a GPU.

## Bottlenecks (top 3, with evidence)

### 1. `kv_cache_kv` materializes the whole cache per layer-step

`forward_chunk` calls `kv_cache_kv` once per layer per DMD step
(profile: 120 calls in a 30-layer bench run — 30 layers x 4 steps).
On the CPU it allocates two fresh `Vec<f32>` (K and V), fills them by
copying the cache row-major, then `upload()` copies each one again.

At steady state the cache is 18,720 rows x 1,536 dims: K and V are
115.0 MiB each, so **one call moves ~460 MiB** (2x fill + 2x upload
copy) — **~55 GiB per chunk, ~386 GiB per 81-frame clip,
~4.6 TiB per 60 s clip** — for data the attention kernel then reads
once and throws away. The wgpu backend has the same shape: it
allocates fresh K/V buffers and runs an elementwise transpose kernel
per call (its cache stores rows as `[rows, heads·head_dim]` while
attention wants `[heads, rows, d]`).

**Fix:** stop materializing. Either store the cache natively in
`[heads, rows, d]` layout (append packs into it — the wgpu `kv_pack`
kernel already exists for append) so `kv_cache_kv` is a view, or give
the attention kernel the row-major cache directly. This is the single
largest structural win in the codebase and it helps every backend.

### 2. Causal attention FLOPs at real scale

~6.05e11 FLOPs per layer-step, 60% of DiT compute (table above), and
the CPU implementation is a scalar triple loop (`attention_impl`).
On a GPU this is the flash-attention kernel's job — the wgpu backend
already has one (`flash` kernel with dense/VSA/causal-masked modes),
but it has never executed anywhere: no GPU on this VM. The masked
pairs (~10%) are skipped in the CPU loop, so the mask does save work;
the cost is the 90% that remain, against a 12-frame cache.

**Fix:** benchmark the existing wgpu flash kernel on a real GPU
first (`bench --backend wgpu --layers 30`); if the causal-masked mode
is slow, the next step is a block-sparse/VSA path for the cached
frames (the `sparse_attention` op already exists for this — it is
currently only exercised by the conformance check, never by the
pipeline).

A related host-side cost in the same path: `forward_chunk` builds the
causal mask as a `Vec<bool>` of `s x rows` — **109.5M booleans ≈
104.5 MiB per (chunk, step) call**, rebuilt 4x per chunk (336 builds
per 60 s clip). It is built once per call and shared across layers,
but never across steps. A bitset representation (13.7 MiB) or a mask
cached across the chunk's 4 steps (temporal indices only change when
the cache evicts) would cut both the allocation and the fill loop.

### 3. TAEHV `conv2d` on CPU (0.4 GFLOPS)

92% of the toy generate time, 35 calls, mean 1.8 ms/call. The
implementation is a direct conv whose inner tap loop goes through a
per-element closure (`read_cat`) with bounds checks on every tap —
no blocking, no im2col, no SIMD. At toy scale this is the whole
story; at real scale decode is <1% of FLOPs, so this only matters
for CPU/offline use — but it is also the reason the debug build took
6.99 s to decode the toy clip.

**Fix:** implicit-GEMM conv (the wgpu backend already ships one —
reuse the formulation, not the shader), or at minimum a blocked
direct conv with hoisted bounds checks. On the GPU path this op is
already a kernel; the CPU fix is for parity runs and debugging only.

### Also measured, not top-3

- **`linear` (65% of toy denoise).** Scalar f32 triple loop, default
  `x86-64` target (SSE2 only, no FMA/AVX2). At real scale it is 38%
  of DiT compute. Fix: `target-cpu=native` / blocked GEMM / SIMD
  `f32x8` on CPU; on GPU it is a tiled kernel (unmeasured here).
- **`head_split` per-row allocation (fixed).** The old code allocated
  `vec![0.0f32; d]` inside the row loop — at real scale, 3 calls x
  4,680 rows x 120 layer-steps x 7 chunks ≈ 12M allocations of 6 KiB
  per short clip. Hoisted to one reused scratch row
  (`crates/lightx2v-cpu/src/lib.rs`): `head_split` 2.45 ms → 1.79 ms
  in `bench --layers 30`, tests still 39/39. No other hot-path
  per-row/per-call allocation remains except `kv_cache_kv` (#1) and
  the mask (#2).
- **`kv_cache_memrope_update` eviction cost.** Each eviction drains
  and splices the 115 MiB K/V vecs (O(tail) memmove per evicted
  frame, per layer, per step). Secondary to #1; fix with a
  ring/compact-once layout when #1 is addressed.

## What is NOT measured

- **No wgpu kernel has executed anywhere in this repo's history.**
  `--backend wgpu` cleanly skips on this VM (no adapter). All GPU
  numbers — kernel times, the flash/causal attention speed, the
  implicit-GEMM conv, end-to-end clip time — are unmeasured.
- **No real 832x480 denoise.** Blocked on RAM on this VM: f32 weights
  5.96 GiB + ~6.6 GiB MemRoPE cache vs 7.9 GiB (OOM 137 in the Studio
  build). The analytic model above is the substitute until a GPU
  machine runs it.
- **No browser execution.** wasm builds (695,933 B) but was never run
  in a browser here; phone/GPU playtest is the gate.
- **No gallery-embed or UMT5 path.** Prompt encoding for real
  generation is a separate workstream (RunPod); nothing in this file
  measures it.
- **No quantize re-run.** Export timings are the Studio build's
  recorded numbers (see above).

## GPU checklist (run on the user's machine)

```sh
# 1. Conformance on the real backend (expect 11/11)
./target/release/lightx2v check --backend wgpu

# 2. Denoise wall time + per-op profile (CPU profile only for now;
#    wgpu has no profile_report() yet — compare wall time against the
#    CPU 64–66 ms at 30 layers and the analytic ~1.2e14 FLOP/chunk)
LIGHTX2V_PROFILE=1 ./target/release/lightx2v bench --backend wgpu --layers 30

# 3. Real short clip (needs web-weights/model + gallery embeds)
./target/release/lightx2v generate --model-dir web-weights/model \
  --embeds web-weights/gallery/<prompt>.safetensors --frames 81 --out clip
#    expect: denoise + decode split printed; compare decode against the
#    6.9e12 FLOP / 0.8%-of-denoise model and the plan's decode peak

# 4. Serve at real scale (watch RSS: 8 kept jobs x 97 MiB of frames)
./target/release/lightx2v serve --model-dir web-weights/model \
  --gallery web-weights/gallery

# 5. Browser: load https://lightx2v.pages.dev on a GPU machine, pick a
#    gallery prompt, watch chunk progress; the f4 weights are 877 MB
#    over ~32 Range requests
```

## Code changes in this session

- `crates/lightx2v-cpu/src/lib.rs`: `LIGHTX2V_PROFILE=1` per-op
  profiler (24 ops, `profile_report()`/`profile_reset()` on the
  `Backend` trait's existing hooks) + `head_split` scratch-row hoist
  (see above).
- `crates/lightx2v-cli/src/main.rs`: print the profile report after
  `bench`/`check`/`generate`; print decode seconds in both `generate`
  paths (`Timings.decode_s` existed but was never set or printed).
- `docs/benchmarks.md`: this file.

Verification after the changes: `cargo test --workspace --offline`
39 passed, 0 failed; `cargo clippy --workspace --offline -- -D
warnings` clean; `cargo fmt --check` clean. Not pushed — local
commit-ready state only.
