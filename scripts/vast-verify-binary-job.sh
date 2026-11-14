#!/bin/bash
# Vast.ai GPU-fix verification job — PREBUILT BINARY version.
# Runs as the instance onstart script. NO rustup, NO cargo, NO crates.io
# on the host: the binary was built on the build VM and published as a
# GitHub release asset (tag verify-runpod-2026-11-14). The host only:
# downloads the binaries + public R2 weights, verifies hashes/sizes,
# runs check/conformance/bench/generate-81 on wgpu with RUST_BACKTRACE=1,
# and echoes a JSON summary to stdout — retrievable via the Vast.ai logs
# API (PUT /instances/request_logs/{id}/). The instance stays up after
# this script exits; the caller destroys it after retrieval. No
# credentials needed: all downloads are public. No SSH is used.
#
# Binaries (GitHub release verify-runpod-2026-11-14):
#   lightx2v-linux-x86_64            sha256 4fa854d06daa8340c09e3fcad27cac20d8d2a134aefad2a7110831fde41f5885
#   conformance-wgpu-linux-x86_64    sha256 c63632837fba626b3b889c62356c1cbb3d90985bc0612e27d0ff92ff2ea5a808
# Results: /root/results/ (summary.json, timings.txt, *.latent, *.rgb,
# _status.json, _done.json or _error.json). The JSON summary is also
# echoed to stdout (tee) so the logs API captures it.
set -ux
export DEBIAN_FRONTEND=noninteractive
export HOME=/root

RESULTS=/root/results
mkdir -p "$RESULTS"

status() {
  printf '{"stage":"%s","detail":"%s"}\n' "$1" "${2:-}" > "$RESULTS/_status.json"
  echo "stage: $1 ${2:-}"
}

# fail <stage> <msg> — write _error.json, echo it, exit 0 (the instance
# stays up after onstart exits; the caller destroys it after retrieval).
fail() {
  printf '{"status":"error","stage":"%s","msg":"%s"}\n' "$1" "$2" > "$RESULTS/_error.json"
  cat "$RESULTS/_error.json"
  echo "BENCH_JOB_DONE"
  exit 0
}

status started "host up, starting binary verification"

# ---- base packages -------------------------------------------------------
# The pytorch image ships curl + python3; libvulkan1 (the wgpu Vulkan
# loader, dlopened at runtime) may be missing — install it if ldconfig
# cannot see it. Without the Vulkan ICD libs, wgpu silently falls back
# to llvmpipe, so the check output below must show a real GPU adapter.
apt-get update -qq || fail setup "apt-get update failed"
apt-get install -y -qq curl python3 ca-certificates || fail setup "apt-get install failed"
ldconfig -p | grep -q libvulkan.so.1 || apt-get install -y -qq libvulkan1 vulkan-tools \
  || fail setup "libvulkan1 install failed"
status deps "base packages installed"

# ---- prebuilt binaries (GitHub release, verify sha256) --------------------
RELEASE=https://github.com/zaitrarrio/lightx2v-rs/releases/download/verify-runpod-2026-11-14
mkdir -p /root/bin
cd /root/bin
curl -sSfL --retry 3 --retry-delay 5 -o lightx2v "$RELEASE/lightx2v-linux-x86_64" \
  || fail download "lightx2v binary download failed"
curl -sSfL --retry 3 --retry-delay 5 -o conformance-wgpu "$RELEASE/conformance-wgpu-linux-x86_64" \
  || fail download "conformance binary download failed"
chmod +x lightx2v conformance-wgpu
echo "4fa854d06daa8340c09e3fcad27cac20d8d2a134aefad2a7110831fde41f5885  lightx2v" | sha256sum -c - \
  || fail download "lightx2v sha256 mismatch"
echo "c63632837fba626b3b889c62356c1cbb3d90985bc0612e27d0ff92ff2ea5a808  conformance-wgpu" | sha256sum -c - \
  || fail download "conformance-wgpu sha256 mismatch"
./lightx2v --version 2>&1 | head -2 || ./lightx2v --help | head -2
status binaries "prebuilt binaries verified"

# ---- conformance fixture script (baked-in CARGO_MANIFEST_DIR path) --------
# The conformance test binary runs scripts/reference/make_fixture.py from
# the path baked in at build time:
#   /home/hatch/workspace/lightx2v-rs/scripts/reference/make_fixture.py
# Recreate exactly that path on the host and fetch the (stdlib-only) script.
# ALSO create the baked-in manifest dir itself
# (/home/hatch/workspace/lightx2v-rs/crates/lightx2v-wgpu): the test
# builds the script path as CARGO_MANIFEST_DIR + "/../../scripts/...",
# and the kernel cannot resolve the `..` components unless the
# crates/lightx2v-wgpu directory EXISTS — without it, python3 fails
# with ENOENT even though the script file is in place (seen on run 3,
# instance 60660688: two fixture tests failed, then the binary hung).
mkdir -p /home/hatch/workspace/lightx2v-rs/crates/lightx2v-wgpu
mkdir -p /home/hatch/workspace/lightx2v-rs/scripts/reference
curl -sSfL --retry 3 --retry-delay 5 \
  -o /home/hatch/workspace/lightx2v-rs/scripts/reference/make_fixture.py \
  https://raw.githubusercontent.com/zaitrarrio/lightx2v-rs/main/scripts/reference/make_fixture.py \
  || fail setup "make_fixture.py download failed"
status fixture "conformance fixture script in place"

# ---- weights (public R2, verify byte counts) ------------------------------
mkdir -p /root/model
cd /root/model
curl -sSfL --retry 3 --retry-delay 5 -o lightx2v-1.3b-f4.safetensors \
  https://pub-53816876523348cf9aba07eeb6f9ccbf.r2.dev/v1/lightx2v-1.3b-f4.safetensors \
  || fail download "f4 weights download failed"
curl -sSfL --retry 3 --retry-delay 5 -o taew2_1.safetensors \
  https://pub-53816876523348cf9aba07eeb6f9ccbf.r2.dev/v1/taew2_1.safetensors \
  || fail download "TAEHV download failed"
curl -sSfL --retry 3 --retry-delay 5 -o fox.safetensors \
  https://pub-53816876523348cf9aba07eeb6f9ccbf.r2.dev/v1/gallery/a-red-fox-trots.safetensors \
  || fail download "fox embed download failed"
stat -c '%n %s' ./*.safetensors | tee -a "$RESULTS/timings.txt"
[ "$(stat -c%s lightx2v-1.3b-f4.safetensors)" -eq 877080928 ] || fail download "f4 size mismatch"
[ "$(stat -c%s taew2_1.safetensors)" -eq 22642902 ] || fail download "TAEHV size mismatch"
[ "$(stat -c%s fox.safetensors)" -eq 8388696 ] || fail download "fox embed size mismatch"
status downloads "weights verified"

# ---- GPU / Vulkan facts ---------------------------------------------------
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | tee -a "$RESULTS/timings.txt" || true
vulkaninfo --summary 2>&1 | head -40 | tee -a "$RESULTS/timings.txt" || true

# ---- timed-run helper: wall + peak RSS (ru_maxrss of the child) -----------
cd /root/bin
cat > /root/run_timed.py <<'PYEOF'
import resource, subprocess, sys, time
t0 = time.time()
p = subprocess.run(sys.argv[1:])
dt = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"WALL={dt:.2f} MAXRSS_MiB={ru.ru_maxrss/1024:.1f} EXIT={p.returncode} CMD={' '.join(sys.argv[1:])}", flush=True)
sys.exit(0)
PYEOF

BIN=/root/bin/lightx2v
run() { # run <tag> <cmd...>: time it, append to timings.txt, never abort the script
  local tag="$1"; shift
  echo "=== $tag ===" | tee -a "$RESULTS/timings.txt"
  LIGHTX2V_PROFILE=1 python3 /root/run_timed.py "$@" 2>&1 | tee -a "$RESULTS/timings.txt"
  echo "=== end $tag ===" | tee -a "$RESULTS/timings.txt"
}

# ---- benchmark sequence ----------------------------------------------------
# RUST_BACKTRACE=1: if generate panics again, the backtrace names the exact
# copy call site (the 2026-11-13 Vast.ai panic had none).
export RUST_BACKTRACE=1
run check-wgpu $BIN check --backend wgpu
# Full wgpu conformance on GPU (3 tests): checks + tiny-DiT rollout vs the
# CPU oracle + TAEHV decode vs the CPU oracle. The rollout exercises the
# whole KV-cache path (put / memrope update / kv) at tiny scale.
# --test-threads=1: the three tests each create a WgpuBackend; creating
# them concurrently hung the binary on run 3 (no output for 9+ min, GPU
# idle). timeout 600: a hang must not block bench/generate.
run wgpu-conformance timeout 600 /root/bin/conformance-wgpu --test-threads=1
run bench-wgpu $BIN bench --backend wgpu --layers 30
run bench-cpu $BIN bench --backend cpu --layers 30
run generate-wgpu-81 timeout 1500 $BIN generate --model-dir /root/model --embeds /root/model/fox.safetensors \
  --backend wgpu --frames 81 --out "$RESULTS/clip-real-wgpu"

# 60 s rollout only if the 81-frame GPU run succeeded and was quick (<5 min).
WGPU81_OK=0
WGPU81_WALL=0
if [ -f "$RESULTS/clip-real-wgpu.rgb" ]; then
  WGPU81_OK=1
  WGPU81_WALL=$(python3 -c "
import re
t = open('$RESULTS/timings.txt').read()
for m in re.finditer(r'WALL=([0-9.]+) MAXRSS_MiB=[0-9.]+ EXIT=0 CMD=(.+generate.+--backend wgpu --frames 81.+)', t):
    print(m.group(1))
" | head -1)
  WGPU81_WALL=${WGPU81_WALL:-0}
fi
if [ "$WGPU81_OK" = 1 ] && python3 -c "exit(0 if float('${WGPU81_WALL:-0}') < 300 else 1)"; then
  run generate-wgpu-1005 timeout 1500 $BIN generate --model-dir /root/model --embeds /root/model/fox.safetensors \
    --backend wgpu --frames 1005 --out "$RESULTS/clip-real-wgpu-60s"
else
  echo "SKIP generate-wgpu-1005 (81-frame ok=$WGPU81_OK wall=$WGPU81_WALL)" | tee -a "$RESULTS/timings.txt"
fi

# ---- summary ----------------------------------------------------------------
ls -la "$RESULTS" | tee -a "$RESULTS/timings.txt"
python3 - "$RESULTS" <<'PYEOF' | tee "$RESULTS/summary.json"
import json, os, re, sys
d = sys.argv[1]
timings = open(os.path.join(d, "timings.txt")).read() if os.path.exists(os.path.join(d, "timings.txt")) else ""
runs = {}
for m in re.finditer(r"WALL=([0-9.]+) MAXRSS_MiB=([0-9.]+) EXIT=(-?\d+) CMD=(.+)", timings):
    runs[m.group(4)] = {
        "wall_s": float(m.group(1)), "maxrss_mib": float(m.group(2)), "exit": int(m.group(3))}
files = {}
for f in sorted(os.listdir(d)):
    p = os.path.join(d, f)
    if os.path.isfile(p):
        files[f] = os.path.getsize(p)
print(json.dumps({"status": "done", "runs": runs, "files": files}, indent=2))
PYEOF
printf '{"status":"done"}\n' > "$RESULTS/_done.json"
echo "BENCH_JOB_DONE"
