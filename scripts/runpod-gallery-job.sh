#!/bin/bash
# No-SSH gallery-embeddings job: runs as a RunPod pod's start command
# (dockerArgs). Clones lightx2v-rs, encodes the 12 gallery prompts with
# UMT5-XXL, and uploads the gallery straight to R2 from the pod.
#
# No SSH is needed at any point: progress is reported by uploading
# _status.json / _done.json / _error.json to the same R2 prefix, and the
# pod exits when this script exits (billing stops with it).
#
# Env required: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT
# (the R2 S3 endpoint, https://<account>.r2.cloudflarestorage.com).
# Nothing in this script ever prints an env value.
set -u
BUCKET="lightx2v-weights"
PREFIX="v1/gallery"
LOG=/root/gallery-job.log
exec > >(tee -a "$LOG") 2>&1
echo "== gallery job starting $(date -u +%FT%TZ) =="

: "${R2_ACCESS_KEY_ID:?missing R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?missing R2_SECRET_ACCESS_KEY}"
: "${R2_ENDPOINT:?missing R2_ENDPOINT}"

# boto3 first: the status reporter below needs it.
python3 -m pip install -q boto3 || { echo "pip install boto3 failed"; exit 1; }

# status <stage> [detail] — upload a small JSON marker to R2.
status() {
  python3 - "$1" "${2:-}" <<'PYEOF'
import json, os, sys
import boto3
stage, detail = sys.argv[1], sys.argv[2]
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  region_name="auto")
body = json.dumps({"stage": stage, "detail": detail}).encode()
s3.put_object(Bucket="lightx2v-weights", Key="v1/gallery/_status.json",
              Body=body, ContentType="application/json")
print(f"status -> {stage}: {detail}")
PYEOF
}

# fail <step> — upload _error.json with the log tail, then exit 1.
fail() {
  local step="$1"
  echo "!! FAILED at $step"
  python3 - "$step" <<'PYEOF'
import json, os, sys
import boto3
step = sys.argv[1]
try:
    with open("/root/gallery-job.log") as f:
        tail = f.read().splitlines()[-60:]
except OSError:
    tail = []
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  region_name="auto")
body = json.dumps({"status": "error", "step": step, "tail": tail}).encode()
s3.put_object(Bucket="lightx2v-weights", Key="v1/gallery/_error.json",
              Body=body, ContentType="application/json")
print(f"error marker uploaded for step {step}")
PYEOF
  exit 1
}

status started "pod up, installing deps"
python3 -m pip install -q torch transformers safetensors sentencepiece \
  || fail "pip-install"
status deps "torch/transformers/safetensors/sentencepiece installed"

git clone --depth 1 https://github.com/zaitrarrio/lightx2v-rs.git /root/lightx2v-rs \
  || fail "git-clone"
status cloned "repo cloned"
cd /root/lightx2v-rs || fail "cd-repo"

status encoding "downloading UMT5-XXL (~11 GB) and encoding 12 prompts"
python3 scripts/encode_prompt.py --prompt-file web/gallery-prompts.txt --out-dir gallery \
  || fail "encode"

status uploading "encode done, uploading gallery to R2"
python3 - <<'PYEOF'
import json, os, pathlib
import boto3
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                  region_name="auto")
gallery = pathlib.Path("gallery")
files = sorted(p for p in gallery.iterdir() if p.is_file())
info = []
for p in files:
    data = p.read_bytes()
    s3.put_object(Bucket="lightx2v-weights", Key=f"v1/gallery/{p.name}", Body=data)
    head = s3.head_object(Bucket="lightx2v-weights", Key=f"v1/gallery/{p.name}")
    assert head["ContentLength"] == len(data), f"size mismatch on {p.name}"
    info.append({"file": p.name, "bytes": len(data)})
    print(f"uploaded {p.name} ({len(data)} B)")
embeds = [f for f in info if f["file"].endswith(".safetensors")]
assert len(embeds) == 12, f"expected 12 embeds, got {len(embeds)}"
assert any(f["file"] == "gallery.json" for f in info), "gallery.json missing"
body = json.dumps({"status": "complete", "prompts": 12, "files": info}).encode()
s3.put_object(Bucket="lightx2v-weights", Key="v1/gallery/_done.json",
              Body=body, ContentType="application/json")
print("done marker uploaded: 12 prompts, %d files" % len(info))
PYEOF
[ $? -ne 0 ] && fail "upload"

echo "== gallery job complete $(date -u +%FT%TZ) =="
exit 0
