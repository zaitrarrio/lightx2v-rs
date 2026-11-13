# Gallery prompt embeddings on RunPod — runbook

Status 2026-11-13: BLOCKED on RunPod access. No RunPod API key exists in the
Secure Vault (`credentials.list`: only claude.ai, google.com,
maximalize.workers.dev, saleshandy.com) and there is no RunPod connector in
`~/workspace/custom-connectors/`. Memory holds no key value.

SECURITY: RunPod emailed 2026-11-05 about suspicious activity on a machine
that hosted the user's pods and asked for new API keys + rotation of all
pod-accessible credentials. No rotation has happened and no user decision is
on record. Do NOT reuse any pre-2026-11-05 key silently. The key for this
job should be a FRESH key created in the RunPod console after that notice
(Settings → API Keys), used once via stdin, never written to a file.

## What the job is

Encode the 12 prompts in `web/gallery-prompts.txt` with UMT5-XXL into
`web-weights/gallery/` (git-ignored, never commit):

- Model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, subfolders
  `tokenizer` + `text_encoder` (see `scripts/encode_prompt.py`, `MODEL`).
  Public on HF — no HF token needed.
- Output per prompt: `<slug>.safetensors`, tensor `prompt_embeds`
  `[512, 4096]` F32, rows past the real token count zeroed; plus
  `gallery.json` = `{"prompts": [{prompt, file, tokens}]}`.
  ~8 MB per file, ~96 MB total.
- Needs ~12 GB RAM (encoder page cache) + ~11 GB model download.
  CPU torch is fine; CUDA optional.

## Pod (cheapest that fits)

RunPod GraphQL API: `POST https://api.runpod.io/graphql`,
`Authorization: Bearer <RUNPOD_API_KEY>`. Create with the
`podFindAndDeployOnDemand` mutation (docs: https://docs.runpod.io/sdks/graphql/manage-pods).

**2026-11-13 field report (6 deploys, all stalled except 1):** the
runbook's image tag `runpod/pytorch:2.9.0-cu1300-ubuntu2404` does NOT
exist — current tags are `runpod/pytorch:2.2.0-rc.229-cu*-torch*-ubuntu2404`
(all RC; old stable tags were removed from the registry). Only 1 of 6
pods ever reported a `runtime` (secure RTX 4090, ~4 min); the rest sat
with `machineId` + `lastStartedAt` set but `runtime: null` for 10–26
min across images (pytorch RC, ubuntu:22.04, nvidia/cuda:12.8.1-runtime),
clouds (community + secure), GPU types (A4000, 4090) and inputs
(with/without `volumeInGb`, with/without `ports: "22"`). A pod
deployed the same way by another agent (`nvidia/cuda:12.8.1-runtime-ubuntu24.04`,
RTX 5090) started in ~3 min — it is a host lottery. **SSH from this
VM is blocked at the egress proxy** (`CONNECT ssh.runpod.io:22` hangs;
`api.runpod.io:443` works), so even a running pod is unreachable from
here until `ssh.runpod.io` is allowlisted. Also: deploy with env
`SSH_PUBLIC_KEY=<this VM's public key>` so the VM key is authorized
(the account's registered key is unknown). Total spend for the 6
stalled attempts was ~$1.50, almost all of it idle waiting time.

**2026-11-13 no-SSH retry — SUCCESS.** The user chose the no-SSH path and
it worked on the third deploy attempt. Deploys 1–2 (secure RTX 4090, then
secure RTX 5090) failed instantly at deploy time with
`SUPPLY_CONSTRAINT` ("no longer any instances available") — no pod
created, no spend. Deploy 3 (community RTX A4000, $0.17/hr,
`runpod/pytorch:2.2.0-rc.229-cu1300-torch291-ubuntu2404`) created pod
`7synm9ys15zuoo`, runtime at ~3 min, `_done.json` at ~11.5 min after
runtime (model download + 12 encodes), terminated immediately after
verification. **Spend for this job ≈ $0.05** (16 min wall × $0.17/hr;
failed deploys cost $0). The contract that made it work, all verified
before deploying: GraphQL `env` is `[{key, value}]`; `dockerArgs` is the
container's command (a bare shell string, treated as CMD — it replaces
the image's Cmd `/start.sh` while the NVIDIA entrypoint execs it, so
the pod runs the job script instead of the default start script and
exits when the script exits). Image config verified from the registry
(Entrypoint `/opt/nvidia/nvidia_entrypoint.sh`, Cmd `/start.sh`).
Job script `scripts/runpod-gallery-job.sh` reports via `_status.json` /
`_done.json` / `_error.json` in `v1/gallery/`; all 13 files verified
publicly (12 embeds × 8,388,696 B, gallery.json 2,722 B; 2 embeds
spot-checked: F32 [512, 4096], rows past real tokens zeroed). The pod
showed desired RUNNING with a stale runtime after the job finished, so
it was terminated explicitly via `podTerminate` and confirmed gone
from `runpod pods`. Note: a sibling agent's `h3-rs-dbg2` pod was
RUNNING at the same time — not this job's, left alone.

- Prefer a CPU pod with >= 24 GB RAM (`cpuTypes` query) if cheaper than GPU.
- Else the cheapest GPU pod with >= 24 GB RAM: community-cloud RTX A5000
  (~$0.27/hr) / L4 (~$0.39/hr) / RTX 4090 (~$0.34/hr) — check live
  `gpuTypes` pricing immediately before creating; prices float.
- Image: `runpod/pytorch:2.9.0-cu130-ubuntu2404` or the current
  `runpod/pytorch:*` image. Expose SSH (default) — no HTTP port needed.
- Expected runtime: model download (~11 GB) + pip install + 12 CPU encodes
  ≈ 30–60 min. Expected spend well under $1 on a $0.27–0.45/hr pod.

## Pod steps

```bash
# on the pod (copy these two files in first; never copy credentials)
pip install torch transformers safetensors sentencepiece
python3 scripts/encode_prompt.py --prompt-file gallery-prompts.txt --out-dir gallery
ls -la gallery/   # expect 12 .safetensors + gallery.json
```

Verify on the pod (spot-check ≥ 2 files + the manifest):

```python
from safetensors import safe_open
import json
g = json.load(open("gallery/gallery.json"))
assert len(g["prompts"]) == 12
for e in g["prompts"][:2]:
    with safe_open(f"gallery/{e['file']}", framework="pt") as f:
        t = f.get_tensor("prompt_embeds")
        assert t.shape == (512, 4096), t.shape
        assert (t[e["tokens"]:] == 0).all(), "rows past real tokens not zeroed"
print("gallery verified:", len(g["prompts"]), "prompts")
```

Download the gallery back to `~/workspace/lightx2v-rs/web-weights/gallery/`
(scp from the pod's SSH endpoint), then TERMINATE the pod immediately
(`podTerminate` mutation) and record final spend.

## R2 upload (files ~8 MB each — under wrangler's 300 MiB cap)

Proven route: `/tmp/cf-wrangler.py` with the surrogate Cloudflare credential
(see `docs/design.md` §10):

```bash
cd ~/workspace/lightx2v-rs
for f in web-weights/gallery/*; do
  python3 /tmp/cf-wrangler.py r2 object put "lightx2v-weights/v1/gallery/$(basename $f)" \
    --file "$f" --remote
done
```

Verify each file publicly (expect 200/206, correct sizes):

```bash
B=https://pub-53816876523348cf9aba07eeb6f9ccbf.r2.dev/v1/gallery
curl -s -o /dev/null -w "gallery.json %{http_code}\n" $B/gallery.json
for f in web-weights/gallery/*.safetensors; do
  curl -s -o /dev/null -w "$(basename $f) %{http_code}\n" -H "Range: bytes=0-1023" \
    "$B/$(basename $f)"
done
```

## Done when

- 12 embeds + gallery.json on R2 under `v1/gallery/`, all public-GET verified.
- Pod terminated, final spend recorded.
- Studio can then run real browser generation end-to-end (still unverified
  until a real GPU browser run — headless WebGPU cannot run on this VM).
