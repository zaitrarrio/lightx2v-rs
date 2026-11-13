"""UMT5-XXL prompt embeddings for `lightx2v generate --embeds` and
`lightx2v serve --gallery`.

lightx2v-rs does not port the 11 GB text encoder yet (docs/design.md,
Phase 3), so prompts are encoded once with transformers and saved:

    pip install torch transformers safetensors sentencepiece
    python scripts/encode_prompt.py "a fox running through snow" fox.safetensors

Matches Diffusers `WanPipeline._get_t5_prompt_embeds`: 512 tokens, the
embeddings past the prompt's real length zeroed. The gallery flow encodes
every prompt in web/gallery-prompts.txt once (see --prompt-file) and
writes one safetensors file per prompt plus gallery.json into the gallery
directory the server and the browser page read:

    python scripts/encode_prompt.py --prompt-file web/gallery-prompts.txt \\
        --out-dir web-weights/gallery

Gallery layout (same as fastwan-rs's gallery, same tensor name):
    gallery/
      gallery.json          {"prompts": [{"prompt": "...", "tokens": 33,
                                          "file": "a-red-fox.safetensors"}]}
      a-red-fox.safetensors tensor "prompt_embeds" [512, 4096], F32, rows
                            past the prompt's real token count zeroed.

Needs ~12 GB of RAM (the encoder page cache) — run it on a machine with
the headroom, not on this VM.
"""
import argparse
import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5EncoderModel

MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"


def encode(tok, enc, dev, prompt):
    t = tok(
        [prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        h = enc(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0].float()
    n = int(t.attention_mask.sum())
    h[n:] = 0
    return h.cpu().contiguous(), n


def slug(prompt, i):
    keep = "".join(c if c.isalnum() or c in " -" else "" for c in prompt.lower())
    words = keep.split()[:4]
    return "-".join(words) or f"prompt-{i}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="a single prompt to encode")
    p.add_argument("out", nargs="?", help="output .safetensors for the single prompt")
    p.add_argument("--prompt-file", help="text file, one prompt per line")
    p.add_argument("--out-dir", help="gallery directory (created; existing prompts kept)")
    args = p.parse_args()

    if args.prompt_file:
        prompts = [
            line.strip()
            for line in pathlib.Path(args.prompt_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not args.out_dir:
            p.error("--prompt-file needs --out-dir")
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    elif args.prompt and args.out:
        prompts = [args.prompt]
        out_dir = None
    else:
        p.error("give one prompt + output file, or --prompt-file + --out-dir")

    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(
        MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(dev).eval()

    if out_dir is None:
        h, n = encode(tok, enc, dev, prompts[0])
        save_file({"prompt_embeds": h}, args.out)
        print(f"{args.out}: [512, {h.shape[1]}], {n} prompt tokens")
        return

    entries = []
    gallery_json = out_dir / "gallery.json"
    if gallery_json.exists():
        entries = json.loads(gallery_json.read_text()).get("prompts", [])
    have = {e["prompt"] for e in entries}
    for i, prompt in enumerate(prompts):
        if prompt in have:
            continue
        name = f"{slug(prompt, i)}.safetensors"
        h, n = encode(tok, enc, dev, prompt)
        save_file({"prompt_embeds": h}, out_dir / name)
        entries.append({"prompt": prompt, "file": name, "tokens": n})
        print(f"{name}: [512, {h.shape[1]}], {n} prompt tokens")
    gallery_json.write_text(json.dumps({"prompts": entries}, indent=2) + "\n")
    print(f"gallery: {len(entries)} prompts in {out_dir}")


if __name__ == "__main__":
    main()
