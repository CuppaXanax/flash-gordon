#!/usr/bin/env python3
"""Extract native-name non-expert GGUF tensors with validated HTTP ranges."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
import urllib.request

REPO = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF"
REV = "06a318de4011598a142ee54bcfec47c6921c958e"
SIZES = {0: (1, 4), 1: (1, 2), 7: (32, 24), 8: (32, 34), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 20: (32, 18), 30: (1, 2)}


def fetch(url, start, size, allow_eof=False):
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                url + ("&" if "?" in url else "?") + f"fg_range={start}-{size}",
                headers={"Range": f"bytes={start}-{start + size - 1}"})
            with urllib.request.urlopen(req, timeout=180) as response:
                if response.status != 206:
                    raise ValueError(f"server ignored range: {response.status}")
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                expected_end = start + size - 1
                if match and allow_eof:
                    expected_end = min(expected_end, int(match[3]) - 1)
                if not match or int(match[1]) != start or int(match[2]) != expected_end:
                    raise ValueError(f"wrong Content-Range: {content_range}")
                data = response.read(size + 1)
                if len(data) != expected_end - start + 1:
                    raise ValueError("short/oversized range")
                return data
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


class Header:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        end = self.pos + n
        if end > len(self.data):
            raise EOFError()
        result = self.data[self.pos:end]
        self.pos = end
        return result

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.read(self.number("Q")).decode("utf-8")

    def value(self, kind):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I",
                   5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind == 8:
            return self.string()
        if kind == 9:
            subtype, count = self.number("I"), self.number("Q")
            for _ in range(count):
                self.value(subtype)
            return None
        return self.number(formats[kind])


def header(url):
    size = 1024 * 1024
    while True:
        if url.startswith("https://"):
            data = fetch(url, 0, size, allow_eof=True)
        else:
            with open(url, "rb") as file:
                data = file.read(size)
        h = Header(data)
        try:
            if h.read(4) != b"GGUF" or h.number("I") != 3:
                raise ValueError("not GGUF v3")
            nt, nk = h.number("Q"), h.number("Q")
            metadata, alignment = [], 32
            for _ in range(nk):
                begin = h.pos
                key, kind = h.string(), h.number("I")
                value = h.value(kind)
                if key == "general.alignment":
                    alignment = value
                if not key.startswith("split."):
                    metadata.append(h.data[begin:h.pos])
            tensors = []
            for _ in range(nt):
                name, dims = h.string(), h.number("I")
                shape = [h.number("Q") for _ in range(dims)]
                kind, offset = h.number("I"), h.number("Q")
                tensors.append(dict(name=name, shape=shape, type=kind, offset=offset))
            return metadata, tensors, (h.pos + alignment - 1) // alignment * alignment, alignment
        except EOFError:
            size *= 2
            if size > 128 * 1024 * 1024:
                raise ValueError("GGUF header exceeds 128 MiB")


def selected(name):
    return ("_exps." not in name and name != "per_layer_token_embd.weight"
            and not name.startswith(("v.", "mm.", "mtp.", "nextn.")))


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--headers-only", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory, metadata, alignment = [], None, None
    for shard in range(1, 9):
        filename = f"BF16/Qwen3.8-Flash-Next-BF16-{shard:05}-of-00008.gguf"
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{filename}"
        md, tensors, base, align = header(url)
        if metadata is None:
            metadata, alignment = md, align
        for t in tensors:
            if not selected(t["name"]):
                continue
            block, nbytes = SIZES[t["type"]]
            count = 1
            for dim in t["shape"]:
                count *= dim
            if count % block:
                raise ValueError(f"invalid block geometry: {t['name']}")
            t.update(url=url, source_offset=base + t["offset"],
                     bytes=count // block * nbytes)
            inventory.append(t)
        print(f"header {shard}/8: {len(tensors)} tensors, {len(inventory)} selected", flush=True)
    names = [t["name"] for t in inventory]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names")
    cursor = 0
    for t in inventory:
        cursor = (cursor + alignment - 1) // alignment * alignment
        t["offset"] = cursor
        cursor += t["bytes"]
    ledger = dict(repo=REPO, revision=REV, tensor_bytes=sum(t["bytes"] for t in inventory),
                  excluded="routed experts, per-layer ngram table, vision, MTP", tensors=inventory)
    args.output.with_suffix(".json").write_text(json.dumps(ledger, indent=2) + "\n")
    if args.headers_only:
        return
    out_header = b"GGUF" + struct.pack("<IQQ", 3, len(inventory), len(metadata)) + b"".join(metadata)
    for t in inventory:
        out_header += (string(t["name"]) + struct.pack("<I", len(t["shape"]))
                       + struct.pack("<" + "Q" * len(t["shape"]), *t["shape"])
                       + struct.pack("<IQ", t["type"], t["offset"]))
    out_header += bytes((-len(out_header)) % alignment)
    base = len(out_header)
    partial = args.output.with_suffix(".partial")
    with partial.open("wb") as out:
        out.write(out_header)
        for i, t in enumerate(inventory):
            out.seek(base + t["offset"])
            digest = hashlib.sha256()
            for off in range(0, t["bytes"], 8 * 1024 * 1024):
                data = fetch(t["url"], t["source_offset"] + off,
                             min(8 * 1024 * 1024, t["bytes"] - off))
                out.write(data)
                digest.update(data)
            t["sha256"] = digest.hexdigest()
            out.flush()
            print(f"{i+1}/{len(inventory)} {t['name']} {t['bytes']}", flush=True)
        os.fsync(out.fileno())
    partial.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    main()
