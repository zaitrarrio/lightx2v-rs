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

## Generation (denoise + decode + rollout)

This section measures the actual generation path — the causal rollout
over chunks, then TAEHV decode — rather than `bench` (one chunk) or
`check`. Same hardware as above (1.5 shared CPUs, 7,934 MiB RAM, no
GPU); release build. Two code changes made this measurement possible
(net +26 lines, tests still 39/39, clippy + fmt clean):

- `generate --synthetic` now honors `--frames` (81 or 1005). It was
  hardcoded to the 9-frame toy shape and ignored the flag, so the
  1005-frame rollout below could not run at all. The earlier
  [`generate --synthetic`](#generate---synthetic-toy-clip-denoise--decode)
  section's 9-frame numbers predate this change and still stand for
  that shape.
- `Timings` records per-chunk wall seconds (`chunks_s`), and both
  `generate` paths print a one-line chunk summary (mean/min/max/
  first/last). The per-chunk time was already measured for the
  pipeline's `Chunk` event; it just was never stored.

### Long rollout, synthetic: `generate --synthetic --frames 1005`

84 chunks of 3 latent frames (252 latent frames → 1005 video frames),
tiny DiT, 64x64. Two runs:

```sh
LIGHTX2V_PROFILE=1 ./target/release/lightx2v generate --synthetic --frames 1005 --out clip1005
```

| run | denoise (84 chunks) | decode (252 latent frames) | wall | peak RSS |
|---|---|---|---|---|
| 1 | 0.642 s | 5.242 s | 5.93 s | 21.2 MiB |
| 2 | 0.693 s | 5.170 s | 5.93 s | 20.6 MiB |

| run | chunk mean | chunk min | chunk max | chunk 1 | chunk 84 |
|---|---|---|---|---|---|
| 1 | 0.008 s | 0.005 s | 0.017 s | 0.005 s | 0.008 s |
| 2 | 0.008 s | 0.004 s | 0.012 s | 0.004 s | 0.010 s |

**Chunk cost is flat across the rollout.** Mean 8 ms/chunk in both
runs; the first chunk is the cheapest (4–5 ms, empty cache) and the
last sits at the mean (8–10 ms). The single 17 ms max in run 1 is a
shared-CPU spike, not a trend — run 2's max is 12 ms. That is the
MemRoPE cache bound working as designed: the cache fills over the
first 4 chunks (12 frames) and then stays bounded, so chunk 84 costs
the same as chunk 5. Per-chunk denoise is 7.6 ms mean; decode is
20.8 ms per latent frame, 5.2 ms per video frame — the same ~22 ms /
latent frame as the 9-frame run, so decode scales linearly with
latent frames, as expected for whole-clip decode.

**Decode is 88% of the 1005-frame generation** (5.2 s of 5.9 s
wall); denoise is 11%. Per-op profile, run 1 (profile total
5,810 ms of 5,930 ms wall):

| op | calls | total | mean/call | share |
|---|---|---|---|---|
| `conv2d` | 2,205 | 5,130.9 ms | 2,327.0 µs | 88.3% |
| `causal_attention` | 672 | 340.1 ms | 506.1 µs | 5.9% |
| `linear` | 4,708 | 245.2 ms | 52.1 µs | 4.2% |
| `download` | 147 | 29.5 ms | 201.0 µs | 0.5% |
| `attention` (cross) | 672 | 15.5 ms | 23.1 µs | 0.3% |
| everything else | — | <45 ms | — | <0.8% |

RSS grows almost linearly, 9.7 → 20.8 MiB across the run — that is
the 12.3 MiB RGB output accumulating during decode, not the rollout
(the latent is preallocated; the MemRoPE cache is bounded).

**Outputs — the first saved test clips this repo has ever
produced.** `clip1005.latent` 258,048 B (64,512 f32 = 252 latent
frames × 256) and `clip1005.rgb` 12,349,440 B (1005 × 64 × 64 × 3,
exact). Both runs wrote identical sizes (`clip1005b.*` likewise).
They are synthetic toy-scale clips (random tiny DiT, zero embeds),
not model output — but they are the first generation artifacts on
disk. `*.latent` was already gitignored; `*.rgb` was added to
`.gitignore` in this session, so neither is committed.

### Real weights, 81 frames, CPU: OOM characterization (one run)

```sh
./target/release/lightx2v generate --model-dir web-weights/model \
  --embeds web-weights/gallery/a-red-fox-trots.safetensors --out clip-real
```

The embed is the public R2 gallery file (8,388,696 B, F32 [512,
4096]); the model dir holds the f4 export + TAEHV, default
`--weights native`. Run **once** under the RSS sampler — one
characterization is enough, and it was done:

| | |
|---|---|
| wall to death | 31.17 s |
| exit | -9 (SIGKILL, kernel OOM killer) |
| peak sampled RSS | 5,664.6 MiB (last sample, 29 s) |
| phase that died | **load** — the log is one line (`backend: cpu`); the `loaded real DiT + TAEHV` line never printed |
| outputs | none (`clip-real.*` do not exist) |

RSS by second: 0.28 GiB at 0 s, 1.75 GiB at 5 s, then a steady
~240 MiB/s climb (1.88 → 5.66 GiB from 13 s to 29 s) — block-by-block
DiT loading, killed in the load tail before TAEHV or any denoise.

**Correction to the earlier record:** the [Quantize](#quantize--export-real-weights--from-the-studio-build-not-re-run)
section says the 81-frame denoise was OOM-killed. This run died
earlier — during load. The load itself is knife-edge on this VM: an
earlier run did complete it in 34.6 s, but this one died at 5.66 GiB
with the OS and shell taking the rest of the 7.9 GiB. Either way the
conclusion stands and is stronger: even a surviving load leaves the
process holding ~6 GiB of f32 weights, and denoise would add the
3,290.6 MiB MemRoPE cache + 1,551.6 MiB activations from the plan —
real 81-frame generation cannot complete on this VM in any phase.
The CPU backend expands the f4 file to f32 on load (5,413 MiB in the
plan's f32 row vs 2,706.5 MiB f16), which is what makes load itself
marginal.

### Serve generation

Nothing new to measure here: the [Serve](#serve-lightx2v-serve---synthetic-release)
section already has the full generation roundtrip (per-job `total_s`
and `decode_s`, 3-job wall 0.61 s). Serve with the real model-dir
was deliberately **not** run — it is the same OOM as the CLI run
above and would add no information.

### Browser generation (not run here — no headless WebGPU)

Known numbers, all from code or `plan`, none executed:

| fact | value |
|---|---|
| `lightx2v plan` largest buffer, 832x480 short clip | 828.8 MiB |
| page size after load | auto-switches to 480×272 (lighter option) |
| progress callback | per-DMD-step ms in the chunk label, per-chunk labels |
| console on load | `lightx2v: WebGPU <adapter> \| <limits>` |
| error surfacing | fixed 2026-11-13 (commit `580c5f5`): wasm errors are real `Error`s now, page falls back to `e.message \|\| e` — the earlier `Failed: undefined` cannot recur |
| gap | `lightx2v-wgpu` does not implement `profile_report()` yet, so browser/GPU runs only get wall-clock step timings for now |

Capture procedure for the user's next attempt — these five lines
produce a profile:

1. Reload https://lightx2v.pages.dev, Load model (f4), pick a gallery prompt.
2. Generate (81 frames, 480×272).
3. Copy the summary line: `Browser (WebGPU) · <w>x<h> · <decoded>/<total> frames · seed <n> · <steps> DMD steps · <s> total.`
4. Copy the console line: `lightx2v: WebGPU <adapter> | <limits>`.
5. If it fails, copy the failure text (`Browser generation failed: …` or `Failed: …`) — it now carries the real reason.

### Generation bottlenecks (top 3)

These are the generation-path bottlenecks specifically; the
general top-3 in [Bottlenecks](#bottlenecks-top-3-with-evidence)
still stand (kv-cache materialization is #1 there too, in
per-layer-step terms).

#### 1. TAEHV decode (`conv2d`) is ~90% of generation wall at every measured scale

88.3% of the 1005-frame profile (5,131 ms / 5,810 ms), 89% of its
wall, 93% of the 9-frame toy clip. 2,205 calls at 2.33 ms mean — a
scalar direct conv at ~0.4 GFLOPS. At real scale decode is <1% of
FLOPs ([Real-scale model](#real-scale-model)), so this bottleneck is
CPU-only: on a GPU it is a kernel, and the fix matters only for
CPU/offline use. **Next optimization:** implicit-GEMM or blocked
direct conv with hoisted bounds checks (same fix as bottleneck #3
in the general list; no chunking win is available — progressive
`decode_chunk` in serve costs the same as whole-clip decode).

#### 2. KV-cache materialization, in generation terms

672 `kv_cache_kv` calls in the 1005-frame run (84 chunks × 4 steps ×
2 layers) — trivial at toy scale (3.2 ms total), but the same call
is ~460 MiB of traffic per layer-step at real scale: 10,080 calls
per 60 s clip ≈ 4.6 TiB of moved data for values the attention
kernel reads once. **Next optimization:** store the cache natively
in `[heads, rows, d]` so the op becomes a view (same fix as
general bottleneck #1).

#### 3. The memory wall: real generation cannot run on this VM

Load alone peaks at 5.7–6.5 GiB against 7.9 GiB (died at 31 s /
5.66 GiB this run; survived once at 34.6 s), and denoise needs a
further ~4.8 GiB on top. This is a capability gap, not a speed
issue: no profiling of real generation is possible here at all.
**Next optimization:** run it on a GPU machine ([GPU
checklist](#gpu-checklist)) — or, for CPU, stop expanding f4 weights
to f32 on load (dequantize per matvec, or stream blocks), which is a
larger change than the <30-line budget for this session.

Not a bottleneck, worth recording: **chunk cost is flat** (table
above) — the MemRoPE 12-frame bound holds across all 84 chunks, so
the rollout has no per-chunk cost growth to optimize away.

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
- **No real 832x480 denoise — characterized, not completed.** One
  careful run ([Generation](#generation-denoise--decode--rollout)):
  died in **load**, not denoise, at 31.17 s / 5,664.6 MiB peak RSS,
  exit -9; an earlier run did complete load in 34.6 s, so load is
  knife-edge, and denoise would need ~4.8 GiB more on top. The
  analytic model above is the substitute until a GPU machine runs it.
- **No browser execution.** wasm builds but was never run in a
  browser here; phone/GPU playtest is the gate. The error-surfacing
  fix is in (commit `580c5f5`) and the capture procedure for the next
  attempt is in [Generation](#generation-denoise--decode--rollout).
- **No real-model test clip.** The only saved clips are synthetic
  toy-scale (`clip1005.latent` / `.rgb`, 1005 frames at 64x64). No
  video from the real weights exists anywhere.
- **Gallery embeds exist now, UMT5 still unmeasured.** The 12
  gallery embeds are live on R2 and one was the `--embeds` arg of
  the OOM run above; the UMT5 encoder itself is still unmeasured.
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

## Code changes in the generation session (2026-11-13)

- `crates/lightx2v-model/src/pipeline.rs`: `Timings` records
  per-chunk wall seconds (`chunks_s`, +3 lines net).
- `crates/lightx2v-cli/src/main.rs`: `generate --synthetic` honors
  `--frames` (was hardcoded to 9 frames) and both `generate` paths
  print a per-chunk summary (`print_chunk_stats`, ~13 lines).
- `.gitignore`: added `*.rgb` (`*.latent` was already covered) so the
  first saved test clips stay out of git.
- `docs/benchmarks.md`: this file's Generation section.

Net diff for the code wiring: +26 lines. Verification after these
changes: `cargo test --workspace --offline` 39 passed, 0 failed;
`cargo clippy --workspace --offline -- -D warnings` clean;
`cargo fmt --check` clean. Not pushed — local commit-ready state
only. The `clip1005*.latent` / `.rgb` outputs are gitignored, not
committed.
