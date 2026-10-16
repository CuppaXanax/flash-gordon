#!/usr/bin/env python3
"""Range-export the shared vocabulary head and one actual next-token embedding."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from mtp_source import DEFAULT_REPO, Source, q8_0


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--embedding", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--token", type=int, default=198)
    a = p.parse_args()
    source = Source(DEFAULT_REPO, "298f94632b784e26a7fe576114f82066689d5baa")
    weights = source.json("model.safetensors.index.json")["weight_map"]
    names = {}
    for kind, suffix in (("head", "lm_head.weight"), ("embedding", "embed_tokens.weight")):
        matches = [n for n in weights if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"ambiguous {kind}: {matches}")
        name = matches[0]
        shard = weights[name]
        length = struct.unpack("<Q", source.read_range(shard, 0, 8))[0]
        if not 2 <= length <= 2 * 1024 * 1024:
            raise ValueError("invalid safetensors header")
        raw = source.read_range(shard, 8, length)
        item = json.loads(raw)[name]
        if item["dtype"] != "BF16" or item["shape"] != [248320, 2560]:
            raise ValueError(f"unexpected shared shape: {item}")
        start, end = item["data_offsets"]
        if end - start != math.prod(item["shape"]) * 2:
            raise ValueError("invalid shared extent")
        names[kind] = dict(name=name, shard=shard, offset=8 + length + start,
                           bytes=end - start, header_sha256=hashlib.sha256(raw).hexdigest())
    if not 0 <= a.token < 248320:
        raise ValueError("embedding token out of range")
    t = names["embedding"]
    raw = source.read_range(t["shard"], t["offset"] + a.token * 2560 * 2, 2560 * 2)
    values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite embedding")
    with a.embedding.open("xb") as out:
        out.write(values.astype("<f4").tobytes())
    t = names["head"]
    digest = hashlib.sha256()
    with a.head.open("xb") as out:
        for start in range(0, t["bytes"], 8 * 1024 * 1024):
            size = min(8 * 1024 * 1024, t["bytes"] - start)
            raw = source.read_range(t["shard"], t["offset"] + start, size)
            digest.update(raw)
            values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")
            if not np.isfinite(values).all():
                raise ValueError("nonfinite shared vocabulary weight")
            out.write(q8_0(values))
            print(f"head source {start + size}/{t['bytes']}", flush=True)
    t["data_sha256"] = digest.hexdigest()
    a.audit.write_text(json.dumps(dict(revision=source.revision, tensors=names,
                                      embedding_token=a.token), indent=2) + "\n")


if __name__ == "__main__":
    main()
