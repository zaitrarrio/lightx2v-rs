"""Build the Prompt Studio static site into web/dist/.

Copies web/studio.html -> web/dist/index.html, the wasm engine
(web/pkg/, if scripts/build-web.sh has run) -> web/dist/pkg/, and
web/gallery-prompts.txt -> web/dist/gallery-prompts.txt, plus a minimal
_headers file. No model weights are ever copied into dist/: the browser
fetches them from the R2 bucket named by --weights (or ?weights= at
runtime), one transformer block per HTTP Range request.

With --weights <base URL>, the __LIGHTX2V_WEIGHTS__ placeholder in
studio.html is replaced by the base URL and web/dist/manifest.json is
written from the local web-weights/ export (sizes from the actual files,
so the page's progress bar and variant picker are honest). Without
--weights the page offers the tiny synthetic model only.
"""

import argparse
import json
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parent.parent
web = root / "web"
dist = web / "dist"

p = argparse.ArgumentParser()
p.add_argument(
    "--weights",
    metavar="BASE_URL",
    help="public R2 base URL the browser streams weights from, e.g. "
    "https://pub-xxx.r2.dev/v1 (writes web/dist/manifest.json)",
)
args = p.parse_args()

shutil.rmtree(dist, ignore_errors=True)
dist.mkdir(parents=True)

html = (web / "studio.html").read_text()
if args.weights:
    html = html.replace("__LIGHTX2V_WEIGHTS__", args.weights.rstrip("/"))
(dist / "index.html").write_text(html)
shutil.copy(web / "gallery-prompts.txt", dist / "gallery-prompts.txt")
if (web / "pkg" / "lightx2v_web_bg.wasm").exists():
    shutil.copytree(web / "pkg", dist / "pkg")
else:
    print("note: web/pkg missing; run scripts/build-web.sh for the in-browser mode")

if args.weights:
    ww = root / "web-weights"
    f4 = ww / "lightx2v-1.3b-f4.safetensors"
    f16 = ww / "lightx2v-1.3b-f16.safetensors"
    tae = ww / "taew2_1.safetensors"
    for f in (f4, f16, tae):
        if not f.exists():
            raise SystemExit(
                f"missing {f}: run `lightx2v quantize` first (see docs/design.md §10)"
            )
    manifest = {
        "model": "lightx2v",
        "name": "LightX2V Wan2.1-T2V-1.3B DMD",
        "variants": {
            "f4": {
                "file": "lightx2v-1.3b-f4.safetensors",
                "bytes": f4.stat().st_size,
                "label": "4-bit",
            },
            "f16": {
                "file": "lightx2v-1.3b-f16.safetensors",
                "bytes": f16.stat().st_size,
                "label": "f16",
            },
        },
        "tae": {"file": "taew2_1.safetensors", "bytes": tae.stat().st_size},
        "gallery": "gallery/gallery.json",
        "sources": {
            "transformer": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            "tae": "fastwan public R2 (taew2_1.safetensors)",
        },
    }
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote manifest.json for", args.weights)

(dist / "_headers").write_text(
    "/*\n  X-Content-Type-Options: nosniff\n"
    "  Referrer-Policy: strict-origin-when-cross-origin\n"
)
print("wrote", dist)
