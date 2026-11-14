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

### Vast.ai (real GPU): first wgpu execution — two bugs found, no clip

2026-11-13, on-demand instance via the Vast.ai API (connector:
`~/workspace/custom-connectors/custom.vast-ai/`). Cheapest offer
meeting the spec at the time (1 GPU, gpu_ram ≥ 16 GiB, rentable,
on-demand, direct ports, disk ≥ 50 GiB): **offer 60331183, NVIDIA
CMP 170HX, 64 GiB VRAM, $0.2948/hr**, disk 50 GiB, host in Latvia,
40,198 MiB CPU RAM. Image `vastai/pytorch:cuda-12.8.1-auto`
(tag verified on Docker Hub first). The job ran from the instance's
onstart script (`scripts/vast-bench-job.sh`, committed with this
section): apt + rustup + `cargo build --release` + public-R2
downloads + the benchmark sequence, results served on port 8080 and
echoed to stdout. Full log retrieved via the Vast.ai logs API
(`PUT /instances/request_logs/{id}/` → `result_url`); see
[Retrieval](#retrieval) for why the other two tiers failed.

| step | value |
|---|---|
| boot (create → `running`) | 279 s (image cached on the host) |
| `cargo build --release` | 31 s (30.16 s) |
| weights downloaded + verified | f4 877,080,928 B, TAEHV 22,642,902 B, fox embed 8,388,696 B — all byte-exact |
| GPU | NVIDIA CMP 170HX, driver 610.43.03, 65,536 MiB VRAM, Vulkan 1.3.275 |
| real DiT + TAEHV load | **7.7 s** (vs 34.6 s CPU load on this VM) |

| run | wall | peak RSS | exit | result |
|---|---|---|---|---|
| `check --backend wgpu` | 0.68 s | 253.1 MiB | 1 | **10/11** — see the sparse failure below |
| `bench --backend wgpu --layers 30` | 1.49 s | 306.1 MiB | 0 | 1.097 s total (load 0.005, prompt 0.154, denoise 0.918) |
| `bench --backend cpu --layers 30` | 0.06 s | 10.0 MiB | 0 | 0.056 s total (same as this VM — single-threaded) |
| `generate --backend wgpu --frames 81` | **778.23 s** | 1,700.8 MiB | **101** | **panicked in a buffer copy — no outputs** |

`check --backend wgpu` in full (first real wgpu kernel execution
anywhere — the dense kernels are correct):

```
PASS linear                                 max_abs_err=0.000e0 tol=1e-6
PASS layer_norm                             max_abs_err=1.192e-7 tol=1e-6
PASS attention                              max_abs_err=1.192e-7 tol=1e-6
PASS causal_attention_mask                  max_abs_err=2.980e-8 tol=1e-5
FAIL sparse==dense@0-sparsity-gate-0        max_abs_err=7.001e-1 tol=1e-5
PASS patchify/unpatchify roundtrip          max_abs_err=0.000e0 tol=1e-6
PASS dmd_step                               max_abs_err=0.000e0 tol=1e-6
PASS quant q8/q4/fp8 roundtrip              max_abs_err=6.552e-2 tol=2e-1
PASS rope origin identity                   max_abs_err=0.000e0 tol=0e0
PASS memrope cache bounded + sinks + ema    max_abs_err=0.000e0 tol=1e-6
PASS history-contrast residual              max_abs_err=0.000e0 tol=1e-6
11 checks, 10 passed, 1 failed
```

**Bug 1 — the VSA sparse path is wrong on GPU.** Dense attention
passes at 1.19e-7, but sparse-vs-dense disagrees by **0.70** — the
sparse kernel's output is essentially wrong, not a precision issue.
This is independent of the generate failure below.

**Bug 2 — the real-scale generate dies in a buffer copy after
778 s of denoise compute.** The GPU sat at 100% utilization for
~12 min (verified via the instance status: `gpu_util` 99.99%,
temp 48 °C), then:

```
thread 'main' panicked at wgpu-30.0.1/src/backend/wgpu_core.rs:2678:18:
wgpu error: Validation Error

Caused by:
  In a CommandEncoder
    Copy at offset 0 for 86274048 bytes would end up overrunning the bounds of the Source buffer of size 57520128
```

The denoise kernels ran fine; the failure is in a copy/readback
path. The copy sites in `lightx2v-wgpu` (`download()`,
`lib.rs:304`/`778`, and `copy()`, `lib.rs:~801`) all assume 4
bytes/element (`b.len * 4`); the panicking copy asked for
86,274,048 B from a 57,520,128 B source — some buffer's `len`
disagrees with its allocation by ~1.5× at real scale. The
per-chunk latent download (`pipeline.rs:236`) is only ~1.2 MB, so
the failing copy is elsewhere in the denoise path; the KV-cache
ops are the prime suspects by code reading (`cache_grow` /
`cache_ema_fold` / `kv_cache_kv`), but that is **not proven** —
pinning the exact call needs `RUST_BACKTRACE=1` on a GPU run.
No "real latent" line ever printed, so denoise never returned;
the 1005-frame run was skipped by the script's gate (81-frame
failed). The `bench --backend wgpu` run printed no per-op profile
(`lightx2v-wgpu` still has no `profile_report()`).

**What this means for the browser failure.** The user's browser
attempt failed fast with no progress event, so the binding-limit
refusal is still the best fit for *that* failure. But these two
bugs are now the next blockers in line: the browser engine uses
the same wgpu backend (`download_async` has the same `len * 4`
assumption, and the KV-cache `copy()` path is shared), so a
browser run that got past the binding limit would hit the copy
panic ~13 min into denoise, and the sparse error would corrupt
the latent even if the copy were fixed. Real generation is
blocked on two independent wgpu correctness bugs, not on speed.

#### Retrieval

All three tiers were attempted; only the logs API worked:

1. **SSH — blocked.** `ssh -p 13236 root@ssh6.vast.ai` dies at
   the egress proxy (`Connection reset by 198.19.0.1 port 3128`),
   same as RunPod. The account's registered keys do not include
   this VM's key either, so SSH was doubly unavailable.
2. **HTTP on the direct port — blocked.** The onstart script did
   start `python3 -m http.server 8080` (pid 6474 in the log) and
   the port was mapped (8080/tcp → HostPort 23934), but `curl`
   from this VM times out — the egress proxy does not allow
   CONNECT to an arbitrary IP:port. No clip files existed to
   fetch anyway (generate failed before writing any).
3. **Logs API — worked.** `PUT /instances/request_logs/{id}/`
   with `{"tail": 300}` returns a `result_url`; a plain GET on
   it returns the full container log (12,798 chars here). Every
   number in this section comes from that log — the script tees
   all output to `timings.txt` and prints `summary.json` to
   stdout, so nothing was lost. The logs API is the retrieval
   channel for future Vast.ai jobs.

#### Attempts and spend

Four creates; three died before the benchmark, all destroyed
in the same run, none left running:

| # | offer / instance | outcome |
|---|---|---|
| 1 | V100 $0.0556/hr, 60548459 | **bad image tag** — `vastai/pytorch:2.2.0-rc.229-…` does not exist (that tag is RunPod's, not Vast.ai's); daemon `manifest unknown`, instance stopped |
| 2 | RTX 3090 $0.1222/hr, 60549331 | **host CDI failure** — `failed to inject CDI devices: unresolvable CDI devices …/gpu=0: unknown`; host-side GPU attach broken |
| 3 | V100 $0.0556/hr, 60549897 | booted in 836 s, then **rustup toolchain download failed** mid-way (`static.rust-lang.org` dropped); the job script was hardened with retries (3× rustup `--profile minimal`, 2× cargo build, 3× curl `--retry`) before the next create |
| 4 | CMP 170HX $0.2948/hr, 60551623 | the run above; boot 279 s, destroyed 22:42 UTC after retrieval |

Total spend ≈ **$0.35** (instance 4: ~58 min × $0.3026/hr ≈
$0.29; the three failed ones ≈ $0.06 combined). The V100 offer
vanished from the bundles between attempts 1 and 4 — offers churn,
so re-check live before creating.

#### Fix verification run 1 — failed before any benchmark (rustup downloads timed out)

2026-11-13 ~23:14–00:39 UTC. Fix commit `8a2a419` (local only, NOT
pushed; VSA sparse + KV-cache copy-panic fixes, on top of `6672108`)
was verified on **instance 60560018, Tesla V100 32 GB, offer
47966236, $0.0672/hr**, image `vastai/pytorch:cuda-12.8.1-auto`.
The onstart wrapper cloned main (`b3bde63`) and applied the fix as
a gzip+base64 patch — `PATCH_OK`, clean apply — then ran the
(patched) `scripts/vast-bench-job.sh`.

**Result: the job never reached a benchmark.** The rustup install
failed after all 3 retries — every attempt timed out downloading
from `static.rust-lang.org` (attempt 1: the rustup-init downloader
timed out; attempts 2–3: the rustc 1.100.0 component download
timed out mid-body). The script's `fail()` wrote
`{"status":"error","stage":"rustup","msg":"rustup install failed after 3 attempts"}`
and exited 0; the container then sat idle (`gpu_util` 0.0,
`cpu_util` 0.0) until the instance was destroyed. No
`check`/`wgpu-conformance`/`bench`/`generate` output exists, so
**the fix itself is neither verified nor refuted** — run 1 of the
2-run cap is consumed with no signal on the two bugs.

This is the second rustup-download failure on Vast.ai (attempt 3
above died the same way); the 3× retry hardening did not save this
run. A future run should pre-bake the Rust toolchain into the
image (or fetch it from a mirror) instead of relying on
`static.rust-lang.org` from the instance at boot.

Two operational notes from this run: the detached poller (pid
1830) died at 23:18 UTC after a single 403 from the logs API and
wrote nothing further — the log was retrieved by a manual poll at
00:39 UTC (the 403 was transient; the same call then worked).
Instance destroyed 00:39 UTC after retrieval, verified gone
(`vast get` → `{"instances": null}`). Spend ≈ **$0.10** (~85 min ×
$0.0672/hr). Run 2 was not started — per the run instructions,
report and stop after a failed run 1.

#### Fix verification run 2 — failed at boot (image pull never completed)

2026-11-13 ~03:02–04:05 UTC. Last of the 2-run cap. Fix commit
`8a2a419` + rustup-hardening commit `9b4156b` (both local only, NOT
pushed; hardening: 5 rustup attempts, curl `--retry 5
--retry-delay 10 --retry-all-errors --max-time 120`, growing
backoff). **Instance 60584719, Tesla V100 32 GB, offer 58421593,
host 381795** (run 1's host 381079 excluded), image
`vastai/pytorch:cuda-12.8.1-auto`. Live rate $0.0806/hr (offer
quoted $0.0689/hr). The fix patch was regenerated from local git
(`6672108..9b4156b`, 18,378 B) and verified to apply cleanly to a
fresh clone of remote main `b3bde63` before the create; the onstart
wrapper (9,785 chars, under the 16 KB limit) was rebuilt with the
new patch.

**Result: the instance never booted.** The image pull on this fresh
host crawled for 62 minutes and never completed — Docker layers
cycled through retrying (`ae9716fa7429: Retrying`, then
`d3d18d7f70db: Download complete`, `15643a6bfa16: Verifying
Checksum`, `72fe3c11b185: Retrying`), each layer taking ~10 min,
and at 04:01 UTC the pull was back to retrying the *first* layer.
Status flapped `loading`/`offline` the whole time; `actual_status`
never reached `running`. The container never started — the logs API
returned `Error response from daemon: No such container:
C.60584719` and no container log exists. The onstart wrapper (and
therefore the patch, the hardened rustup, and every benchmark) never
ran. **The fix is still neither verified nor refuted.**

Instance destroyed 04:05 UTC after retrieval of the final status,
verified gone (`vast get` → `{"instances": null}`); the detached
poller (pid 7236) was stopped. Spend ≈ **$0.08** (~63 min ×
$0.0806/hr). Combined run-1 + run-2 spend ≈ $0.18; combined with
the first benchmark run ≈ $0.53 total across all Vast.ai work.

Lesson for any future verification: both failed runs died on host
egress, not on the code — run 1 on `static.rust-lang.org`, run 2 on
the Docker registry. Pick a host with the image already cached
(boot in ~5 min when cached, vs 60+ min and counting here) or use a
smaller image; pre-baking the Rust toolchain into the image would
remove the second failure mode as well. The two GPU bugs remain
open with no new evidence either way.

#### Fix verification, RunPod attempt — no pod ever got a runtime (7 pods, 0 runtimes)

2026-11-14 ~04:08–05:30 UTC (2026-11-13 ~22:08–23:30 CST). User
redirected here after both Vast.ai verification runs died on host
egress. Same fix commit `8a2a419` (local only) + hardened rustup
commit `9b4156b`. Job script `scripts/runpod-verify-job.sh`
(9,188 B) and the fix patch (17,093 B, `git diff
6672108..8a2a419`, 5 files) were pushed to remote main
(`4def248a`, via the git-data API — blobs → tree with base_tree →
commit → ref update; NOT `push-local`, which would have swept
untracked `.vast-verify/` into the tree). The patch was verified to
apply cleanly to a fresh clone of remote main `b3bde63` before any
deploy. The job script (adapted from `scripts/vast-bench-job.sh`)
starts an HTTP server on port 8080 serving `/root/results` first,
then apt + hardened rustup, clones remote main, downloads and
applies the patch, `cargo build --release`, downloads f4/TAEHV/fox
embed from public R2 with byte-count checks, then `check --backend
wgpu` (want 11/11), `cargo test --release --test conformance wgpu_`
(want 3/3), `bench --backend wgpu/cpu --layers 30`, and `generate
--backend wgpu --frames 81` with `RUST_BACKTRACE=1` (want a
completed `clip-real-wgpu.rgb`/`.latent`), writing `summary.json`
+ `_done.json`/`_error.json` and sleeping 1800 s for retrieval.

**Result: no pod ever got a runtime, so the job script never ran
— not once.** `podFindAndDeployOnDemand` returned
`desiredStatus: RUNNING` with a `machineId` and `lastStartedAt`
set, but `runtime` stayed `null` for every pod, through waits of
12–25 min each. All pods were terminated and verified gone
(`runpod get` → `pod: null`); `runpod pods` shows no live pods.

| # | pod | offer / image | outcome |
|---|---|---|---|
| 1 | `7ms4745abzoxl8` | secure RTX 4090 $0.89/hr, pytorch | no runtime in ~14 min, terminated |
| 2 | `2nf30ai04owanb` | secure RTX 5090 ~$1.39/hr, pytorch | no runtime in ~14 min, terminated |
| 3 | `ua44mchkt19tqf` | community RTX 3090 Ti $0.27/hr, pytorch | no runtime in ~25 min (extended wait — the image pull was suspected), terminated |
| 4 | `or0fsgkcxhtwrm` | community RTX A5000 $0.16/hr, `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | no runtime in ~12 min, terminated |
| 5 | `nguaartyewevnk` | community RTX A4000 $0.17/hr, pytorch | no runtime in ~14 min, terminated |

Two extra pods (`dv971lcy6xpaer` secure RTX 4000 Ada,
`o9jh6i15oqsq6g` secure A40) were created by accident when three
deploys were batched in one shell command and terminated within
~1 min — never batch multiple deploys in one command; each one
spends. Eleven further deploys failed instantly with
`SUPPLY_CONSTRAINT` (no pod created, $0): community A5000, A4000
(×2), A4500, V100-PCIE-16GB, 3090, 4080, 4090; secure A4000,
A4500, A5000. Community cloud was dry; secure 4090/5090 created
but stalled.

Three hypotheses were tested and narrowed: (a) **image pull** —
pod 3 got the longest wait (25 min) and pod 4 used the much
smaller CUDA image, both still stalled, so a slow pull alone does
not explain it; (b) **deploy input** — the gallery job (pod
`7synm9ys15zuoo`, same `podFindAndDeployOnDemand` shape, same
pytorch image, community A4000, runtime in ~3 min) proves the
shape works, and the gallery agent's session record confirms it;
(c) **platform outage** — RunPod status pages (all regions) showed
up, no outages in 24 h. What remains is the host lottery: public
write-ups match exactly — `desiredStatus: RUNNING` with no runtime
means the host is alive while the inner container never reports
(failed image pull, host disk full, or a bad host just freed by a
prior tenant); the fix there is terminate and re-create, switching
to SECURE after two failures. Tonight that did not help either:
5 waited-out pods across both clouds, 0 runtimes. **The fix is
still neither verified nor refuted — a third straight attempt
died before the code could run.**

**Retrieval channel — never tested.** No pod got a runtime, so
the HTTP proxy URL (`https://<pod-id>-8080.proxy.runpod.net/`)
was never exercised; SSH from this VM is blocked (same as
Vast.ai) and a web search confirmed **pod logs are NOT available
via the GraphQL or REST API** — so the HTTP proxy is the only
retrieval channel for a no-SSH RunPod job, and it remains
unproven. If a future pod gets a runtime, test the proxy URL
early, before the long generate.

Spend ≈ **$0.72** (4090 ~$0.21 + 5090 ~$0.32 + 3090 Ti ~$0.11 +
A4000 ~$0.04 + A5000 ~$0.03 + the two 1-min pods ~$0.01), within
the ~$2 cap. No second RunPod run was started — the attempt stops
here for the parent's call. Combined across all three verification
attempts (Vast.ai ×2 + RunPod): ≈ **$0.90** spent, zero benchmark
signal, the two GPU bugs still open. The lesson from all three:
pre-bake the toolchain + deps into the image (or use a host with
the image cached) and keep trying cheap community offers — the
failure mode is always the host, never the code so far.

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

- **wgpu kernels HAVE now executed — and found two bugs.**
  First real execution was 2026-11-13 on Vast.ai
  ([Vast.ai](#vastai-real-gpu-first-wgpu-execution--two-bugs-found-no-clip)):
  dense kernels are correct (attention 1.19e-7), but the VSA sparse
  path is wrong by 0.70 and the real-scale generate dies in a buffer
  copy after 778 s of denoise compute. Both are correctness bugs, not
  speed. The per-op GPU profile is still unmeasured — `lightx2v-wgpu`
  has no `profile_report()`, so GPU runs only give wall-clock totals.
- **No real 832x480 denoise has completed on any backend.** The GPU
  run above computed for 778 s at 100% GPU and then panicked in a
  copy before returning a latent; the CPU run died in load. The
  analytic model above is still the substitute for a finished run.
- **No browser execution.** wasm builds but was never run in a
  browser here; phone/GPU playtest is the gate. The error-surfacing
  fix is in (commit `580c5f5`) and the capture procedure for the next
  attempt is in [Generation](#generation-denoise--decode--rollout).
  Note the Vast.ai finding: even past the binding limit, the browser
  would hit the same copy panic and sparse error, since it shares
  the wgpu backend.
- **No real-model test clip.** The only saved clips are synthetic
  toy-scale (`clip1005.latent` / `.rgb`, 1005 frames at 64x64). No
  video from the real weights exists anywhere.
- **Gallery embeds exist now, UMT5 still unmeasured.** The 12
  gallery embeds are live on R2 and one was the `--embeds` arg of
  the Vast.ai generate run; the UMT5 encoder itself is still unmeasured.
- **No quantize re-run.** Export timings are the Studio build's
  recorded numbers (see above).

## GPU checklist (run on the user's machine)

Items 1–3 are DONE on Vast.ai (2026-11-13, CMP 170HX) — see
[Vast.ai](#vastai-real-gpu-first-wgpu-execution--two-bugs-found-no-clip)
for the numbers. What is left is the fix-then-rerun loop and the
two items never attempted:

```sh
# 1. Conformance on the real backend — DONE, 10/11 (sparse fails by 0.70)
./target/release/lightx2v check --backend wgpu

# 2. Denoise wall time — DONE, 1.097 s for the tiny 30-layer bench
#    (no per-op GPU profile: wgpu has no profile_report() yet)
LIGHTX2V_PROFILE=1 ./target/release/lightx2v bench --backend wgpu --layers 30

# 3. Real short clip — DONE as far as it goes: 778 s of denoise, then a
#    buffer-copy panic, no clip. Re-run AFTER the copy + sparse fixes,
#    with RUST_BACKTRACE=1 if it fails again.
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

## Code changes in the Vast.ai session (2026-11-13)

- `scripts/vast-bench-job.sh` (new): the onstart job script that
  ran on the Vast.ai instance — apt + rustup (3 retries,
  `--profile minimal`) + `cargo build --release` (2 retries) +
  public-R2 downloads (byte-count verified, curl `--retry 3`) +
  check / bench / generate sequence + results HTTP server on 8080
  + JSON summary to stdout. No Rust code changed in this session.
- `docs/benchmarks.md`: the Vast.ai section above; "What is NOT
  measured" and the GPU checklist updated (items 1–3 done, with
  the two wgpu bugs as the next blockers).
- `~/workspace/custom-connectors/custom.vast-ai/CONNECTOR.md`:
  create / poll / destroy / logs-API calls corrected to their
  verified shapes (see that file).

Verification after these changes: `cargo test --workspace --offline`
39 passed, 0 failed; `cargo clippy --workspace --offline -- -D
warnings` clean; `cargo fmt --check` clean. Not pushed — local
commit only.
