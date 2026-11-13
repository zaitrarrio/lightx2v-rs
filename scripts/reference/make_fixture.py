#!/usr/bin/env python3
"""Generate the tiny safetensors fixture for the model conformance test.

Pure stdlib (the safetensors format is an 8-byte header length, a JSON
header, then raw little-endian bytes) — no torch, no numpy. Weights are
deterministic pseudo-random values from `random.Random(name + seed)`, so
regenerating the fixture reproduces it exactly.

Writes: <out>/tiny.safetensors with every tensor of
`WanConfig::tiny()`'s manifest, as F32, plus the tiny TAEHV manifest.
"""
import json
import random
import struct
import sys
from pathlib import Path

# (name, shape) — mirrors lightx2v_core::weights::wan_manifest(tiny())
# and tae_manifest(tiny()); keep in sync with the Rust manifests.
TINY = dict(
    patch=2, heads=2, head_dim=12, in_channels=4, out_channels=4,
    text_dim=16, text_len=8, freq_dim=16, ffn_dim=40, layers=2,
)


def manifest():
    d = TINY["heads"] * TINY["head_dim"]
    p = TINY["patch"]
    v = [
        ("patch_embedding.weight", [d, TINY["in_channels"], 1, p, p]),
        ("patch_embedding.bias", [d]),
        ("condition_embedder.time_embedder.linear_1.weight", [d, TINY["freq_dim"]]),
        ("condition_embedder.time_embedder.linear_1.bias", [d]),
        ("condition_embedder.time_embedder.linear_2.weight", [d, d]),
        ("condition_embedder.time_embedder.linear_2.bias", [d]),
        ("condition_embedder.time_proj.weight", [6 * d, d]),
        ("condition_embedder.time_proj.bias", [6 * d]),
        ("condition_embedder.text_embedder.linear_1.weight", [d, TINY["text_dim"]]),
        ("condition_embedder.text_embedder.linear_1.bias", [d]),
        ("condition_embedder.text_embedder.linear_2.weight", [d, d]),
        ("condition_embedder.text_embedder.linear_2.bias", [d]),
        ("scale_shift_table", [1, 2, d]),
        ("proj_out.weight", [p * p * TINY["out_channels"], d]),
        ("proj_out.bias", [p * p * TINY["out_channels"]]),
    ]
    for l in range(TINY["layers"]):
        pre = f"blocks.{l}"
        for a in ("attn1", "attn2"):
            for m in ("to_q", "to_k", "to_v", "to_out.0"):
                v.append((f"{pre}.{a}.{m}.weight", [d, d]))
                v.append((f"{pre}.{a}.{m}.bias", [d]))
            v.append((f"{pre}.{a}.norm_q.weight", [d]))
            v.append((f"{pre}.{a}.norm_k.weight", [d]))
        v.append((f"{pre}.norm2.weight", [d]))
        v.append((f"{pre}.norm2.bias", [d]))
        v.append((f"{pre}.ffn.net.0.proj.weight", [TINY["ffn_dim"], d]))
        v.append((f"{pre}.ffn.net.0.proj.bias", [TINY["ffn_dim"]]))
        v.append((f"{pre}.ffn.net.2.weight", [d, TINY["ffn_dim"]]))
        v.append((f"{pre}.ffn.net.2.bias", [d]))
        v.append((f"{pre}.scale_shift_table", [1, 6, d]))
    # Tiny TAEHV manifest (mirrors tae_manifest(TaeConfig::tiny())).
    nf = [8, 6, 4, 4]
    tg = [1, 2, 2]
    v.append(("decoder.1.weight", [nf[0], 4, 3, 3]))
    v.append(("decoder.1.bias", [nf[0]]))
    for st in range(3):
        base = 3 + 6 * st
        for m in range(3):
            pp = f"decoder.{base + m}"
            v.append((f"{pp}.conv.0.weight", [nf[st], 2 * nf[st], 3, 3]))
            v.append((f"{pp}.conv.0.bias", [nf[st]]))
            v.append((f"{pp}.conv.2.weight", [nf[st], nf[st], 3, 3]))
            v.append((f"{pp}.conv.2.bias", [nf[st]]))
            v.append((f"{pp}.conv.4.weight", [nf[st], nf[st], 3, 3]))
            v.append((f"{pp}.conv.4.bias", [nf[st]]))
        v.append((f"decoder.{base + 4}.conv.weight", [nf[st] * tg[st], nf[st], 1, 1]))
        v.append((f"decoder.{base + 5}.weight", [nf[st + 1], nf[st], 3, 3]))
    v.append(("decoder.22.weight", [3, nf[3], 3, 3]))
    v.append(("decoder.22.bias", [3]))
    return v


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = {}
    blobs = []
    offset = 0
    for name, shape in manifest():
        n = 1
        for s in shape:
            n *= s
        rng = random.Random(f"lightx2v-fixture:{name}")
        # Small values, norm weights near 1 (like the Synthetic source).
        if name.endswith(("norm_q.weight", "norm_k.weight", "norm2.weight")):
            vals = [1.0 + rng.uniform(-0.1, 0.1) for _ in range(n)]
        else:
            fan_in = shape[1] if len(shape) > 1 else 1
            a = 1.0 / fan_in ** 0.5
            vals = [rng.uniform(-a, a) for _ in range(n)]
        raw = struct.pack(f"<{n}f", *vals)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, sort_keys=True).encode()
    # Header length must be a multiple of 8; pad the JSON with spaces.
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    with open(out / "tiny.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)
    print(f"wrote {out / 'tiny.safetensors'} ({offset} bytes of tensors, "
          f"{len(header)} tensors)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
