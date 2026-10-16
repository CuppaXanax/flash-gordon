#!/usr/bin/env python3
"""Restore one original-Q8 rank from authenticated mixed spans and dense GGUF."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from mtp_format_audit import gguf_header, load_manifest

ORIGINAL = "35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb"
MIXED = "0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252"
DENSE = "918a6175bf6082863b2306c125802970164cbe9777847997e5c1c51aa3096684"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(args):
    original, mixed = load_manifest(args.original), load_manifest(args.mixed)
    if original["sha256"] != ORIGINAL or mixed["sha256"] != MIXED:
        raise ValueError("unrecognized original/mixed manifest")
    if digest(args.dense) != DENSE:
        raise ValueError("dense original-Q8 source hash mismatch")
    before = {row["name"]: row for row in mixed["records"]}
    if set(before) != {row["name"] for row in original["records"]}:
        raise ValueError("original/mixed tensor inventory differs")
    with args.dense.open("rb") as stream:
        sources = {row["name"]: row for row in gguf_header(stream.read(16 << 20))["records"]}
    target = [row for row in original["records"] if row["rank"] == args.rank]
    raw_path = args.output.with_suffix(".raw")
    if not target or args.output.exists() or raw_path.exists():
        raise ValueError("invalid rank or output/scratch already exists")
    high = max(row["offset"] + row["bytes"] for row in target)
    high = (high + 4095) & ~4095
    with args.output.open("xb") as stream:
        stream.truncate(high)
    copied = cooked = routers = 0
    try:
        with args.source.open("rb") as source, args.output.open("r+b") as destination:
            for row in target:
                old = before[row["name"]]
                for field in ("dims", "shape", "rank", "layer", "expert", "kind"):
                    if old[field] != row[field]:
                        raise ValueError(f"changed tensor identity: {row['name']}")
                same = all(old[field] == row[field] for field in ("type", "layout", "bytes", "sha256"))
                if same:
                    source.seek(old["offset"])
                    destination.seek(row["offset"])
                    remaining = row["bytes"]
                    check = hashlib.sha256()
                    while remaining:
                        chunk = source.read(min(8 << 20, remaining))
                        if not chunk:
                            raise ValueError("truncated mixed rank")
                        destination.write(chunk)
                        check.update(chunk)
                        remaining -= len(chunk)
                    if check.hexdigest() != row["sha256"]:
                        raise ValueError(f"mixed span hash mismatch: {row['name']}")
                    copied += 1
                elif row["type"] == 0 and old["type"] == 30 and row["layout"] == old["layout"] == 0:
                    source.seek(old["offset"])
                    raw = source.read(old["bytes"])
                    if hashlib.sha256(raw).hexdigest() != old["sha256"]:
                        raise ValueError("BF16 router source hash mismatch")
                    expanded = b"".join(struct.pack("<I", value[0] << 16)
                                        for value in struct.iter_unpack("<H", raw))
                    if hashlib.sha256(expanded).hexdigest() != row["sha256"]:
                        raise ValueError("router is not lossless against original-Q8")
                    destination.seek(row["offset"])
                    destination.write(expanded)
                    routers += 1
                else:
                    tensor = sources.get(row["name"])
                    if not tensor or tensor["type"] != row["type"] or tensor["shape"] != row["shape"]:
                        raise ValueError(f"missing exact original-Q8 tensor: {row['name']}")
                    destination.flush()
                    with args.dense.open("rb") as dense:
                        dense.seek(tensor["file_offset"])
                        raw = dense.read(tensor["bytes"])
                    if len(raw) != tensor["bytes"]:
                        raise ValueError("truncated dense Q8 source")
                    raw_path.write_bytes(raw)
                    subprocess.run([str(args.placer), str(args.original), row["name"],
                                    str(raw_path), str(args.output)], check=True)
                    raw_path.unlink()
                    cooked += 1
        with args.output.open("rb") as stream:
            for row in target:
                stream.seek(row["offset"])
                check = hashlib.sha256()
                remaining = row["bytes"]
                while remaining:
                    chunk = stream.read(min(8 << 20, remaining))
                    if not chunk:
                        raise ValueError("truncated restored rank")
                    check.update(chunk)
                    remaining -= len(chunk)
                if check.hexdigest() != row["sha256"]:
                    raise ValueError(f"original-Q8 readback mismatch: {row['name']}")
        print(json.dumps(dict(kind="q8_rank_restored", rank=args.rank,
                              spans=len(target), copied=copied, cooked=cooked,
                              lossless_routers=routers, trained_spans_sha_exact=True,
                              original_manifest_sha256=ORIGINAL, mixed_manifest_sha256=MIXED,
                              dense_gguf_sha256=DENSE, file_sha256=digest(args.output))))
    finally:
        raw_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "mixed", "dense", "source", "output", "placer"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int, choices=range(8))
    restore(parser.parse_args())
