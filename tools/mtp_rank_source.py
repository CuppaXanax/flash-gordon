#!/usr/bin/env python3
"""Reconstruct authenticated, unchanged rank tensors using bounded HTTP ranges."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

from mtp_source import Source


def reconstruct(coverage, manifest, placer, directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = Source(coverage["repo"], coverage["revision"])
    if source.revision != coverage["revision"]:
        raise ValueError("source revision changed")
    receipt = directory / "public-source.jsonl"
    if receipt.exists():
        raise ValueError("refusing to replace an existing reconstruction receipt")
    chunk = 16 * 1024 * 1024
    with receipt.open("x", encoding="utf-8") as log, (
            concurrent.futures.ThreadPoolExecutor(max_workers=4)) as pool:
        for entry in coverage["records"]:
            if not entry["same_format"]:
                continue
            target, original = entry["target"], entry["source"]
            raw = directory / "source-tensor.raw"
            digest = hashlib.sha256()
            with raw.open("xb") as output:
                for start in range(0, original["bytes"], 4 * chunk):
                    requests = [(original["file_offset"] + offset,
                                 min(chunk, original["bytes"] - offset))
                                for offset in range(start, min(
                                    start + 4 * chunk, original["bytes"]), chunk)]
                    for data in pool.map(
                            lambda request: source.read_range(
                                original["shard"], *request), requests):
                        output.write(data)
                        digest.update(data)
            rank = directory / f"rank-{target['rank']:02d}.fgw"
            result = subprocess.run([str(placer), str(manifest), target["name"],
                                     str(raw), str(rank)], check=True,
                                    capture_output=True, text=True)
            proof = json.loads(result.stdout)
            if not proof["manifest_tensor_sha_exact"]:
                raise ValueError("cooked tensor did not match manifest")
            proof.update(source=original, source_sha256=digest.hexdigest(),
                         manifest_sha256=target["sha256"])
            log.write(json.dumps(proof) + "\n")
            log.flush()
            raw.unlink()
            print(target["name"], target["bytes"], "SHA exact", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--placer", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    reconstruct(json.loads(args.coverage.read_text()), args.manifest,
                args.placer.resolve(), args.directory)


if __name__ == "__main__":
    main()
