#!/bin/bash
# lightx2v Vast.ai GPU benchmark job — runs as the instance onstart script.
# No SSH, no credentials: clones the public repo, downloads weights from the
# public R2 URLs, runs check/bench/generate on wgpu (+ tiny CPU bench), and
# serves the results over HTTP on port 8080 while echoing a JSON summary to
# stdout (retrievable via the Vast.ai logs API).
#
# Usage: passed as the `onstart` string of PUT /asks/{offer_id}/.
# Results: /root/results/ (summary.json, timings.txt, *.latent, *.rgb,
# _done.json or _error.json). Always exits 0 so the instance stays up
# for retrieval; the caller destroys it.
set -ux
export DEBIAN_FRONTEND=noninteractive
export HOME=/root
export PATH="$HOME/.cargo/bin:$PATH"

RESULTS=/root/results
mkdir -p "$RESULTS"

fail() {
  printf '{"status":"error","stage":"%s","msg":"%s"}\n' "$1" "$2" > "$RESULTS/_error.json"
  cat "$RESULTS/_error.json"
  # keep the HTTP server up for retrieval, then exit 0
  exit 0
}

# ---- base packages -------------------------------------------------------
apt-get update -qq || fail setup "apt-get update failed"
apt-get install -y -qq curl git build-essential pkg-config libvulkan1 vulkan-tools python3 ca-certificates \
  || fail setup "apt-get install failed"

# ---- Rust ----------------------------------------------------------------
# Retry: the host's egress to static.rust-lang.org can drop mid-download.
if ! command -v cargo >/dev/null 2>&1; then
  RUSTUP_OK=0
  for i in 1 2 3; do
    echo "rustup attempt $i"
    if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable; then
      RUSTUP_OK=1; break
    fi
    sleep 10
  done
  [ "$RUSTUP_OK" = 1 ] || fail rustup "rustup install failed after 3 attempts"
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version || fail rustup "cargo not on PATH after install"

# ---- repo + build ---------------------------------------------------------
cd /root
[ -d lightx2v-rs ] || git clone --depth 1 https://github.com/zaitrarrio/lightx2v-rs \
  || fail clone "git clone failed"
cd lightx2v-rs
git log --oneline -1
BUILD_T0=$(date +%s)
BUILD_OK=0
for i in 1 2; do
  echo "cargo build attempt $i"
  if cargo build --release; then BUILD_OK=1; break; fi
  sleep 10
done
[ "$BUILD_OK" = 1 ] || fail build "cargo build --release failed"
BUILD_S=$(( $(date +%s) - BUILD_T0 ))
echo "BUILD_SECONDS=$BUILD_S" | tee -a "$RESULTS/timings.txt"

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

# ---- GPU / Vulkan facts ---------------------------------------------------
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | tee -a "$RESULTS/timings.txt" || true
vulkaninfo --summary 2>&1 | head -40 | tee -a "$RESULTS/timings.txt" || true

# ---- results HTTP server (port 8080, mapped via env "-p 8080:8080") -------
cd "$RESULTS"
nohup python3 -m http.server 8080 >/root/http.log 2>&1 &
echo "HTTP server on 8080 pid $!" | tee -a "$RESULTS/timings.txt"
cd /root/lightx2v-rs

# ---- timed-run helper: wall + peak RSS (ru_maxrss of the child) -----------
cat > /root/run_timed.py <<'PYEOF'
import resource, subprocess, sys, time
t0 = time.time()
p = subprocess.run(sys.argv[1:])
dt = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"WALL={dt:.2f} MAXRSS_MiB={ru.ru_maxrss/1024:.1f} EXIT={p.returncode} CMD={' '.join(sys.argv[1:])}", flush=True)
sys.exit(0)
PYEOF

BIN=./target/release/lightx2v
run() { # run <tag> <cmd...>: time it, append to timings.txt, never abort the script
  local tag="$1"; shift
  echo "=== $tag ===" | tee -a "$RESULTS/timings.txt"
  LIGHTX2V_PROFILE=1 python3 /root/run_timed.py "$@" 2>&1 | tee -a "$RESULTS/timings.txt"
  echo "=== end $tag ===" | tee -a "$RESULTS/timings.txt"
}

# ---- benchmark sequence ----------------------------------------------------
# NOTE: a full real-scale CPU generate is NOT run: the analytic projection
# (~170 h denoise per 81-frame clip, docs/benchmarks.md) makes it infeasible.
# CPU comparison is the tiny synthetic bench only.
run check-wgpu $BIN check --backend wgpu
run bench-wgpu $BIN bench --backend wgpu --layers 30
run bench-cpu $BIN bench --backend cpu --layers 30
run generate-wgpu-81 $BIN generate --model-dir /root/model --embeds /root/model/fox.safetensors \
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
  run generate-wgpu-1005 $BIN generate --model-dir /root/model --embeds /root/model/fox.safetensors \
    --backend wgpu --frames 1005 --out "$RESULTS/clip-real-wgpu-60s"
else
  echo "SKIP generate-wgpu-1005 (81-frame ok=$WGPU81_OK wall=$WGPU81_WALL)" | tee -a "$RESULTS/timings.txt"
fi

# ---- summary ----------------------------------------------------------------
ls -la "$RESULTS" | tee -a "$RESULTS/timings.txt"
python3 - "$RESULTS" "$BUILD_S" <<'PYEOF' | tee "$RESULTS/summary.json"
import json, os, re, sys
d, build_s = sys.argv[1], int(sys.argv[2])
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
print(json.dumps({"status": "done", "build_s": build_s, "runs": runs, "files": files}, indent=2))
PYEOF
printf '{"status":"done"}\n' > "$RESULTS/_done.json"
echo "BENCH_JOB_DONE"
