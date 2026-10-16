#!/usr/bin/env python3
"""Reproduce an unchanged real-pack expert tensor without touching the ring."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import struct

from mtp_format_audit import PINNED_MANIFEST, gguf_header, load_manifest
from mtp_source import Source, get_json

REPO = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF"
REVISION = "7e3bfc316b880fefeb049596f11c49d6a18e05fb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor", default="blk.2.ffn_down_exps.weight.rank1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if manifest["sha256"] != PINNED_MANIFEST:
        raise ValueError("not the sealed mix-04 manifest")
    matches = [record for record in manifest["records"] if record["name"] == args.tensor]
    if len(matches) != 1:
        raise ValueError("ambiguous/missing real tensor")
    target = matches[0]
    if (target["type"], target["layout"], target["shape"]) != (8, 5, [640, 2560, 512]):
        raise ValueError("expected the production cooked Q8 down slab")
    source = Source(REPO, REVISION)
    if source.revision != REVISION:
        raise ValueError("source revision changed")
    listing = get_json(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}/UD-Q4_K_XL")
    name = re.sub(r"\.rank\d+$", "", args.tensor)
    selected = []
    for item in listing:
        shard = item["path"]
        if not shard.endswith(".gguf"):
            continue
        first = source.read_range(shard, 0, 24)
        if first[:8] != b"GGUF\x03\0\0\0":
            raise ValueError("expected GGUF v3")
        if struct.unpack_from("<Q", first, 8)[0] == 0:
            continue
        raw = source.read_range(shard, 0, min(item["size"], 16 * 1024 * 1024))
        header = gguf_header(raw)
        for record in header["records"]:
            if record["name"] == name:
                selected.append(dict(record, shard=shard, header_sha256=header["header_sha256"],
                                     shard_bytes=item["size"], source_lfs=item.get("lfs")))
    if len(selected) != 1:
        raise ValueError("missing/duplicate source tensor")
    tensor = selected[0]
    if any(tensor[key] != target[key] for key in ("type", "shape", "bytes")):
        raise ValueError("source/real-pack tensor geometry differs")
    if tensor["file_offset"] + tensor["bytes"] > tensor["shard_bytes"]:
        raise ValueError("source tensor exceeds shard")
    chunk = 8 * 1024 * 1024
    starts = list(range(0, tensor["bytes"], chunk))
    digest = hashlib.sha256()
    with args.output.open("xb") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for first in range(0, len(starts), 4):
                batch = starts[first:first + 4]
                requests = [pool.submit(source.read_range, tensor["shard"],
                                        tensor["file_offset"] + start,
                                        min(chunk, tensor["bytes"] - start)) for start in batch]
                for start, future in zip(batch, requests):
                    raw = future.result()
                    output.write(raw)
                    digest.update(raw)
                print(f"{min(tensor['bytes'], (batch[-1] + chunk))}/{tensor['bytes']}", flush=True)
    result = dict(repo=REPO, revision=REVISION, manifest_sha256=manifest["sha256"],
                  manifest_tensor=target, source_tensor=tensor,
                  raw_sha256=digest.hexdigest(),
                  required_cooked_sha256=target["sha256"],
                  identity="Not qualified until C cooker verifies full cooked slab SHA against manifest.")
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
