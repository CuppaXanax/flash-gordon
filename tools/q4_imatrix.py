#!/usr/bin/env python3
"""Range-fetch dense imatrix sums/counts without downloading expert tensors."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
from q4_source import fetch, header, string
from q4_quantize import target

URL = ("https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/"
       "766911a6b7369840a91dbcd95f9f997acaab6cd6/imatrix_unsloth.gguf_file")


def main():
    source = Path("dense-bf16.gguf")
    _, dense, _, _ = header(str(source))
    wanted = {t["name"] for t in dense if target(t, False) in (12, 13, 14, 20)}
    metadata, tensors, base, alignment = header(URL)
    selected = [t for t in tensors if any(
        t["name"].endswith(suffix) and t["name"][:-len(suffix)] in wanted
        for suffix in (".in_sum2", ".counts"))]
    if not selected:
        raise ValueError("no matching dense imatrix tensors")
    spans = []
    cursor = 0
    for t in selected:
        if t["type"] != 0:
            raise ValueError("imatrix must contain F32 sums/counts")
        count = 1
        for dim in t["shape"]:
            count *= dim
        t["bytes"] = count*4
        t["source_offset"] = base+t["offset"]
        cursor = (cursor+alignment-1)//alignment*alignment
        t["offset"] = cursor
        cursor += t["bytes"]
        start, end = t["source_offset"], t["source_offset"]+t["bytes"]
        if spans and start >= spans[-1][1] and start-spans[-1][1] < alignment:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    h = b"GGUF"+struct.pack("<IQQ", 3, len(selected), len(metadata))+b"".join(metadata)
    for t in selected:
        h += (string(t["name"])+struct.pack("<I", len(t["shape"]))
              +struct.pack("<"+"Q"*len(t["shape"]), *t["shape"])
              +struct.pack("<IQ", t["type"], t["offset"]))
    h += bytes((-len(h)) % alignment)
    partial = Path("dense-imatrix.partial")
    with ThreadPoolExecutor(max_workers=8) as pool, partial.open("wb") as out:
        out.write(h)
        def download(span):
            start, end = span
            return start, fetch(URL, start, end-start)
        for i, (start, raw) in enumerate(pool.map(download, spans)):
            for t in selected:
                off = t["source_offset"]-start
                if 0 <= off and off+t["bytes"] <= len(raw):
                    out.seek(len(h)+t["offset"])
                    out.write(raw[off:off+t["bytes"]])
            print(f"imatrix ranges {i+1}/{len(spans)}", flush=True)
    path = Path("dense-imatrix.gguf")
    partial.replace(path)
    report = dict(source=URL, ranges=len(spans), tensors=len(selected),
                  bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
