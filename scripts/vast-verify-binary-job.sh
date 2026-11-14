#!/bin/bash
# Vast.ai GPU-fix verification job — PREBUILT BINARY version.
# Runs as the instance onstart script. NO rustup, NO cargo, NO crates.io
# on the host: the binaries were built on the build VM and published as a
# GitHub release asset (tag verify-fixes-2026-11-14). The host only:
# downloads the binaries + public R2 weights, verifies hashes/sizes,
# runs check/conformance/bench/generate-81 on wgpu with RUST_BACKTRACE=1,
# and echoes a JSON summary to stdout — retrievable via the Vast.ai logs
# API (PUT /instances/request_logs/{id}/). The instance stays up after
# this script exits; the caller destroys it after retrieval. No
# credentials needed: all downloads are public. No SSH is used.
#
# Binaries (GitHub release verify-fixes-2026-11-14 — built from LOCAL
# commit 3871e79, NOT from this tag's tree; see the release notes):
#   lightx2v-linux-x86_64            sha256 059b79e05a60272a3937ba533a7f000a169bd3299c107f8169bcb4bf2427658d
#   conformance-wgpu-linux-x86_64    sha256 5ae0040874711849de5af0780553982c493f2af5b52ceb77b3eec1efbfa1c5e5
#   upstream-parity.py               (D10 parity harness, 39,601 B)
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
apt-get install -y -qq curl python3 ca-certificates git || fail setup "apt-get install failed"
ldconfig -p | grep -q libvulkan.so.1 || apt-get install -y -qq libvulkan1 vulkan-tools \
  || fail setup "libvulkan1 install failed"
status deps "base packages installed"

# ---- prebuilt binaries (GitHub release, verify sha256) --------------------
RELEASE=https://github.com/zaitrarrio/lightx2v-rs/releases/download/verify-fixes-2026-11-14
mkdir -p /root/bin
cd /root/bin
curl -sSfL --retry 3 --retry-delay 5 -o lightx2v "$RELEASE/lightx2v-linux-x86_64" \
  || fail download "lightx2v binary download failed"
curl -sSfL --retry 3 --retry-delay 5 -o conformance-wgpu "$RELEASE/conformance-wgpu-linux-x86_64" \
  || fail download "conformance binary download failed"
chmod +x lightx2v conformance-wgpu
echo "059b79e05a60272a3937ba533a7f000a169bd3299c107f8169bcb4bf2427658d  lightx2v" | sha256sum -c - \
  || fail download "lightx2v sha256 mismatch"
echo "5ae0040874711849de5af0780553982c493f2af5b52ceb77b3eec1efbfa1c5e5  conformance-wgpu" | sha256sum -c - \
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

# ---- nvidia-smi VRAM sampler (D6 gate: peak vs plan, alongside the ----
# ---- CLI's "device live bytes" peak line) --------------------------------
# Samples memory.used every 5 s into $1 (max so far); start before the
# first generate, stop before the summary. Background loop, killed by PID.
smi_start() {
  ( while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      if [ -n "$used" ]; then
        prev=$(cat "$RESULTS/smi_peak.txt" 2>/dev/null || echo 0)
        [ "$used" -gt "$prev" ] && echo "$used" > "$RESULTS/smi_peak.txt"
      fi
      sleep 5
    done ) &
  echo $! > "$RESULTS/smi_pid.txt"
}
smi_stop() {
  [ -f "$RESULTS/smi_pid.txt" ] && kill "$(cat "$RESULTS/smi_pid.txt")" 2>/dev/null || true
  peak=$(cat "$RESULTS/smi_peak.txt" 2>/dev/null || echo "?")
  echo "nvidia-smi peak VRAM used during generates: ${peak} MiB" | tee -a "$RESULTS/timings.txt"
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
smi_start
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
smi_stop

# ---- Phase C: upstream parity A/B (D1/D2/D3/D4 fidelity gate) --------------
# Needs >=16 GB host RAM (torch side is ~10 GB: 5.3 GB f32 weights +
# 12-frame KV cache + MemRoPE q copy). Best effort: any failure records
# PARITY_SKIP and the job continues to the summary. The harness script
# is a release asset (not on remote main — the fix commits are local).
# Layout must mirror the repo: <root>/scripts/upstream-parity.py and
# <root>/research/history-forcing-upstream/dmd (clone at 9d69c1e, the
# exact rev the harness was validated against).
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "parity: host RAM ${RAM_GB} GiB" | tee -a "$RESULTS/timings.txt"
set -o pipefail  # from here on, a failure in any piped stage must set PARITY_OK=0
if [ "${RAM_GB:-0}" -ge 16 ]; then
  PARITY_OK=1
  mkdir -p /root/parity/repo/scripts /root/parity/repo/research
  cd /root/parity/repo/scripts
  curl -sSfL --retry 3 --retry-delay 5 -o upstream-parity.py \
    "$RELEASE/upstream-parity.py" || PARITY_OK=0
  if [ "$PARITY_OK" = 1 ]; then
    cd /root/parity/repo/research
    git clone --depth 1 https://github.com/ziplab/History-Forcing.git history-forcing-upstream 2>&1 | tail -2 || PARITY_OK=0
    if [ "$PARITY_OK" = 1 ]; then
      cd history-forcing-upstream
      git fetch --depth 1 origin 9d69c1e5da2beb219e635f4400b8cb494759ba06 2>&1 | tail -1
      git checkout FETCH_HEAD 2>&1 | tail -1 || git checkout 9d69c1e5da2beb219e635f4400b8cb494759ba06 2>&1 | tail -1 || PARITY_OK=0
      cd /root/parity/repo
      git -C research/history-forcing-upstream rev-parse HEAD 2>&1 | tee -a "$RESULTS/timings.txt" || true
    fi
  fi
  if [ "$PARITY_OK" = 1 ]; then
    python3 -c "import torch, safetensors, numpy" 2>&1 | tee -a "$RESULTS/timings.txt" \
      || pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu safetensors numpy 2>&1 | tail -2 \
      || PARITY_OK=0
  fi
  if [ "$PARITY_OK" = 1 ]; then
    cd /root/parity
    curl -sSfL --retry 3 --retry-delay 5 -o f16.safetensors \
      https://pub-53816876523348cf9aba07eeb6f9ccbf.r2.dev/v1/lightx2v-1.3b-f16.safetensors \
      || PARITY_OK=0
    [ "$PARITY_OK" = 1 ] && [ "$(stat -c%s f16.safetensors)" -eq 2979733280 ] || PARITY_OK=0
  fi
  if [ "$PARITY_OK" = 1 ]; then
    cd /root/parity/repo
    echo "=== parity-inputs ===" | tee -a "$RESULTS/timings.txt"
    timeout 300 python3 scripts/upstream-parity.py inputs --out-dir /root/parity/data 2>&1 | tee -a "$RESULTS/timings.txt" || PARITY_OK=0
  fi
  if [ "$PARITY_OK" = 1 ]; then
    echo "=== parity-rust ===" | tee -a "$RESULTS/timings.txt"
    timeout 900 /root/bin/lightx2v parity-chunk --weights /root/parity/f16.safetensors \
      --embeds /root/parity/data/prompt_embeds.safetensors --chunks 7 --out /root/parity/rust 2>&1 | tee -a "$RESULTS/timings.txt" || PARITY_OK=0
  fi
  if [ "$PARITY_OK" = 1 ]; then
    echo "=== parity-torch ===" | tee -a "$RESULTS/timings.txt"
    timeout 1500 python3 scripts/upstream-parity.py upstream --weights /root/parity/f16.safetensors \
      --embeds /root/parity/data/prompt_embeds.safetensors --chunks 7 --out /root/parity/torch 2>&1 | tee -a "$RESULTS/timings.txt" || PARITY_OK=0
  fi
  if [ "$PARITY_OK" = 1 ]; then
    echo "=== parity-compare ===" | tee -a "$RESULTS/timings.txt"
    timeout 300 python3 scripts/upstream-parity.py compare \
      --rust /root/parity/rust.bin --torch /root/parity/torch.bin 2>&1 | tee -a "$RESULTS/timings.txt" || PARITY_OK=0
  fi
  if [ "$PARITY_OK" = 1 ]; then
    echo "PARITY_DONE" | tee -a "$RESULTS/timings.txt"
  else
    echo "PARITY_SKIP (setup or a stage failed — see above)" | tee -a "$RESULTS/timings.txt"
  fi
else
  echo "PARITY_SKIP (host RAM ${RAM_GB} GiB < 16 GiB)" | tee -a "$RESULTS/timings.txt"
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
