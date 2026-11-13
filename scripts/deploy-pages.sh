#!/usr/bin/env bash
# Build web/dist and deploy it to the Cloudflare Pages project `lightx2v`
# (https://lightx2v.pages.dev). Needs CLOUDFLARE_API_TOKEN (Pages: Edit) and
# CLOUDFLARE_ACCOUNT_ID in the environment. The project must already exist;
# create it once with `wrangler pages project create lightx2v`.
#
# dist/ holds only the studio page, the wasm engine and the gallery prompts —
# NEVER model weights. The browser streams weights from the public R2 bucket
# named by --weights (or ?weights=<base URL> at runtime), one transformer
# block per HTTP Range request, into the browser cache. The hosted studio
# drives the engine on the visitor's own machine.
#
#   scripts/deploy-pages.sh --weights https://pub-xxx.r2.dev/v1
#
# Upload the weights first (see docs/design.md §10 for the exact command);
# build-site.py --weights reads the local web-weights/ export to write
# web/dist/manifest.json with the real file sizes.
#
# This script does NOT run a deploy by itself in CI: it only builds dist/.
# The `wrangler pages deploy` line at the end is the manual step.
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    *) echo "unknown arg: $1 (usage: $0 [--weights <base URL>])" >&2; exit 2 ;;
  esac
done
scripts/build-web.sh
if [ -n "$WEIGHTS" ]; then
  python3 scripts/build-site.py --weights "$WEIGHTS"
else
  python3 scripts/build-site.py
fi
echo "dist/ ready. To deploy (needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID):"
echo "  npx -y wrangler@4 pages deploy web/dist --project-name lightx2v --branch main --commit-dirty=true"
