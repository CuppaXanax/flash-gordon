#!/usr/bin/env python3
"""Reconstruct sealed mix-04 weights from pinned public GGUF and its exact overlay."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from mtp_format_audit import PINNED_MANIFEST, gguf_header, load_manifest
from mtp_real_tensor import REPO, REVISION
from mtp_source import Source, get_json

OVERLAY_SHA = "e742dbf593f03c5c608de229329755e7cb19138e7b1c4031632eaf49f1acf4d1"


def metadata(manifest, directory, overlay):
    target = load_manifest(manifest)
    if target["sha256"] != PINNED_MANIFEST:
        raise ValueError("requires sealed mix-04 manifest")
    with overlay.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != OVERLAY_SHA:
            raise ValueError("overlay identity changed")
        file.seek(0)
        mixed = {t["name"]: t for t in gguf_header(file.read(16*1024*1024))["records"]}
    source = Source(REPO, REVISION)
    if source.revision != REVISION:
        raise ValueError("public source revision changed")
    tensors = {}
    header_saved = False
    for item in get_json(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}/UD-Q4_K_XL"):
        shard = item["path"]
        if not shard.endswith(".gguf"):
            continue
        head = source.read_range(shard, 0, 24)
        metadata_only = not struct.unpack_from("<Q", head, 8)[0]
        raw = source.read_range(shard, 0, min(item["size"], 16*1024*1024))
        if metadata_only or not header_saved:
            (directory/"public-header.gguf").write_bytes(raw)
            header_saved = True
        if metadata_only:
            continue
        for tensor in gguf_header(raw)["records"]:
            if tensor["name"] in tensors:
                raise ValueError("duplicate public tensor")
            tensors[tensor["name"]] = dict(tensor, shard=shard)
    records = []
    for t in target["records"]:
        if t["kind"] not in (1, 2, 3):
            continue
        name = re.sub(r"\.rank\d+$", "", t["name"])
        original = tensors.get(name)
        same = original and all(original[k] == t[k] for k in ("type", "shape"))
        selected = original if same else mixed.get(name)
        if not selected or any(selected[k] != t[k] for k in ("type", "shape")):
            raise ValueError(f"no exact raw source for {t['name']}")
        records.append(dict(target=t, source=selected, overlay=not same))
    result = dict(manifest_sha256=target["sha256"], repo=REPO, revision=REVISION,
                  overlay_sha256=OVERLAY_SHA, records=records)
    (directory/"source-map.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Source map", len(records), "records; overlay", sum(r["overlay"] for r in records), flush=True)


def reconstruct(manifest, directory, overlay, placer, rank):
    data = json.loads((directory/"source-map.json").read_text())
    source = Source(REPO, REVISION)
    if source.revision != data["revision"]:
        raise ValueError("source revision changed")
    if load_manifest(manifest)["sha256"] != data["manifest_sha256"]:
        raise ValueError("manifest identity changed")
    chunk = 16*1024*1024
    rows = [r for r in data["records"] if
            (r["target"]["kind"] == 3 if rank == "ngram" else r["target"]["rank"] == int(rank))]
    receipt = directory/f"source-{rank}.jsonl"
    completed = {}
    if receipt.exists():
        completed = {r["tensor"]: r for r in map(json.loads, receipt.read_text().splitlines())}
    with receipt.open("a", encoding="utf-8") as log, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in rows:
            t, s = record["target"], record["source"]
            if t["name"] in completed:
                continue
            raw = directory/("ngram.iq4nl" if rank == "ngram" else "tensor.raw")
            digest = hashlib.sha256()
            with raw.open("wb") as output:
                if record["overlay"]:
                    with overlay.open("rb") as input:
                        input.seek(s["file_offset"])
                        for start in range(0, s["bytes"], chunk):
                            block = input.read(min(chunk, s["bytes"]-start))
                            if not block:
                                raise ValueError("short overlay")
                            output.write(block); digest.update(block)
                else:
                    for begin in range(0, s["bytes"], 4*chunk):
                        requests = [(s["file_offset"]+offset, min(chunk, s["bytes"]-offset))
                                    for offset in range(begin, min(begin+4*chunk, s["bytes"]), chunk)]
                        for block in pool.map(lambda pair: source.read_range(s["shard"], *pair), requests):
                            output.write(block); digest.update(block)
            if rank == "ngram":
                if digest.hexdigest() != t["sha256"]:
                    raise ValueError("complete n-gram tensor SHA differs from sealed pack")
                padded = (t["bytes"]+4095) & ~4095
                with raw.open("r+b") as output:
                    output.truncate(padded)
                proof = dict(tensor=t["name"], bytes=t["bytes"], file_bytes=padded,
                             manifest_tensor_sha_exact=True)
            else:
                result = subprocess.run([str(placer), str(manifest), t["name"], str(raw),
                                         str(directory/f"rank-{int(rank):02d}.fgw")],
                                        check=True, capture_output=True, text=True)
                proof = json.loads(result.stdout)
                raw.unlink()
            proof.update(raw_sha256=digest.hexdigest(), source=s, overlay=record["overlay"])
            log.write(json.dumps(proof)+"\n");log.flush()
            print(t["name"], t["bytes"], "SHA exact", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("metadata", "ngram", *map(str, range(8))))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--placer", type=Path)
    a = p.parse_args();a.directory.mkdir(parents=True, exist_ok=True)
    if a.operation == "metadata":
        metadata(a.manifest, a.directory, a.overlay)
    else:
        if not a.placer:
            p.error("--placer required")
        reconstruct(a.manifest, a.directory, a.overlay, a.placer.resolve(), a.operation)


if __name__ == "__main__":
    main()
