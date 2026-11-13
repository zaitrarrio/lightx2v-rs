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
