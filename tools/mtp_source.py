#!/usr/bin/env python3
"""Audit or export only MTP tensors; never download complete HF shards."""

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import urllib.parse
import urllib.request

DEFAULT_REPO = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated"


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


class Source:
    def __init__(self, repo, revision):
        self.repo = repo
        info = get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}")
        self.revision = info["sha"]
        self.base = f"https://huggingface.co/{repo}/resolve/{self.revision}/"

    def json(self, name):
        return get_json(self.base + name)

    def read_range(self, shard, start, size):
        if size <= 0 or size > 16 * 1024 * 1024:
            raise ValueError("range must be between 1 byte and 16 MiB")
        end = start + size - 1
        url = self.base + shard + f"?fg_range={start}-{end}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match or (
                            int(match[1]), int(match[2])) != (start, end):
                        raise ValueError("server ignored or changed the requested range")
                    data = response.read(size + 1)
                    if len(data) != size:
                        raise ValueError("short or oversized range response")
                    return data
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def audit(self):
        config = self.json("config.json")
        index = self.json("model.safetensors.index.json")
        selected = {k: v for k, v in index["weight_map"].items()
                    if k.startswith("mtp.")}

        def header(shard):
            length = struct.unpack("<Q", self.read_range(shard, 0, 8))[0]
            if not 2 <= length <= 2 * 1024 * 1024:
                raise ValueError("invalid safetensors header size")
            raw = self.read_range(shard, 8, length)
            tensors = json.loads(raw)
            result = []
            for name in sorted(k for k in selected if selected[k] == shard):
                item = tensors[name]
                shape = item["shape"]
                offsets = item["data_offsets"]
                if item["dtype"] != "BF16" or (
                        offsets[1] - offsets[0] != math.prod(shape) * 2):
                    raise ValueError(f"unexpected dtype or size for {name}")
                result.append(dict(name=name, shard=shard, dtype=item["dtype"],
                                   shape=shape, bytes=offsets[1] - offsets[0],
                                   file_offset=8 + length + offsets[0],
                                   header_sha256=hashlib.sha256(raw).hexdigest()))
            return result

        tensors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for items in pool.map(header, sorted(set(selected.values()))):
                tensors.extend(items)
        return dict(repo=self.repo, revision=self.revision, config=config,
                    total_source_tensors=len(index["weight_map"]),
                    tensors=sorted(tensors, key=lambda x: x["name"]),
                    mtp_bf16_bytes=sum(t["bytes"] for t in tensors))


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def q8_0(values):
    import numpy as np
    blocks = values.reshape(-1, 32)
    scale = np.max(np.abs(blocks), axis=1) / np.float32(127)
    inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
    rounded = blocks * inverse[:, None]
    quant = np.copysign(np.floor(np.abs(rounded) + np.float32(.5)), rounded)
    result = np.empty((len(blocks), 34), dtype=np.uint8)
    result[:, :2] = scale.astype("<f2").view(np.uint8).reshape(-1, 2)
    result[:, 2:] = np.clip(quant, -127, 127).astype(np.int8).view(np.uint8)
    return result.tobytes()


def pack(source, audit, destination):
    import numpy as np
    if not audit["tensors"]:
        raise ValueError("source has no MTP tensors")
    entries, offset = [], 0
    for tensor in audit["tensors"]:
        shape = tensor["shape"]
        if len(shape) > 1 and shape[-1] % 32:
            raise ValueError(f"Q8_0 row is not block aligned: {tensor['name']}")
        count = math.prod(shape)
        ggml_type, size = (0, count * 4) if len(shape) == 1 else (8, count // 32 * 34)
        entries.append((tensor, ggml_type, size, offset))
        offset = (offset + size + 31) & ~31
    metadata = (string("general.architecture") + struct.pack("<I", 8) + string("qwen4exp") +
                string("general.name") + struct.pack("<I", 8) + string("MTP overlay") +
                string("flash_gordon.mtp.source_revision") + struct.pack("<I", 8) +
                string(audit["revision"]))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(entries), 3) + metadata
    for tensor, ggml_type, _, offset in entries:
        shape = tensor["shape"]
        header += string(tensor["name"]) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", ggml_type, offset)
    header += b"\0" * (-len(header) % 32)
    # Exclusive creation prevents overwriting a qualified overlay.
    with Path(destination).open("xb") as output:
        output.write(header)
        for tensor, ggml_type, size, offset in entries:
            output.seek(len(header) + offset)
            digest = hashlib.sha256()
            written = 0
            for start in range(0, tensor["bytes"], 8 * 1024 * 1024):
                count = min(8 * 1024 * 1024, tensor["bytes"] - start)
                raw = source.read_range(tensor["shard"], tensor["file_offset"] + start, count)
                digest.update(raw)
                values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"nonfinite source weights: {tensor['name']}")
                cooked = values.astype("<f4").tobytes() if ggml_type == 0 else q8_0(values)
                output.write(cooked)
                written += len(cooked)
            if written != size:
                raise ValueError("export tensor size mismatch")
            tensor["data_sha256"] = digest.hexdigest()
            print(f"{tensor['name']}: {size} bytes", flush=True)
    # Raw HF norm gammas are retained, including Gemma's zero-centred values.
    # The executor must apply the architecture's +1 exactly once.
    audit["export"] = dict(format="GGUF Q8_0 matrices / F32 vectors",
                           raw_norm_gammas=True, bytes=Path(destination).stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pack", type=Path)
    args = parser.parse_args()
    source = Source(args.repo, args.revision)
    audit = source.audit()
    if args.pack:
        pack(source, audit, args.pack)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"{len(audit['tensors'])} MTP tensors, {audit['mtp_bf16_bytes']} BF16 bytes, "
          f"revision {audit['revision']}")


if __name__ == "__main__":
    main()
