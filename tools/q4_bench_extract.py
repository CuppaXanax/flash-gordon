#!/usr/bin/env python3
"""Extract the real dense kernel benchmark matrices without copying unused tensors."""
import argparse
import json
from pathlib import Path
import struct
from q4_source import header, string, SIZES

SHAPES = ((2560,10240),(2560,6144),(6144,2560),(2560,12288),
          (10240,320),(320,10240),(640,2560),(2560,640),(2560,512))


def extract(source, output, q8=False, reference=None):
    _, tensors, base, alignment = header(str(source))
    chosen, cursor = [], 0
    selections = []
    metadata = [string("general.alignment")+struct.pack("<II",4,alignment)]
    if reference:
        metadata, required, _, alignment = header(str(reference))
        available = {t["name"]: t for t in tensors}
        for r in required:
            t = available.get(r["name"])
            if not t or t["shape"] != r["shape"] or t["type"] not in (8,12,13,20):
                raise ValueError(f"missing overlay matrix: {r['name']}")
            selections.append(t)
    else:
        for shape in SHAPES:
            kind = 8 if q8 else 20 if shape[0] % 256 else 12
            candidates = [t for t in tensors if tuple(t["shape"]) == shape and t["type"] == kind]
            if not candidates:
                raise ValueError(f"missing real benchmark matrix {shape} type {kind}")
            selections.append(candidates[0])
    for selected in selections:
        t = dict(selected)
        t["source_offset"] = base+t["offset"]
        block, size = SIZES[t["type"]]
        t["bytes"] = t["shape"][0]*t["shape"][1]//block*size
        cursor = (cursor+alignment-1)//alignment*alignment
        t["offset"] = cursor
        cursor += t["bytes"]
        chosen.append(t)
    h = b"GGUF"+struct.pack("<IQQ",3,len(chosen),len(metadata))+b"".join(metadata)
    for t in chosen:
        h += string(t["name"])+struct.pack("<IQQIQ",2,*t["shape"],t["type"],t["offset"])
    h += bytes((-len(h)) % alignment)
    with source.open("rb") as src, output.open("xb") as dst:
        dst.write(h)
        for t in chosen:
            src.seek(t["source_offset"])
            dst.seek(len(h)+t["offset"])
            left = t["bytes"]
            while left:
                raw = src.read(min(left,8<<20))
                if not raw: raise ValueError("short source matrix")
                dst.write(raw);left -= len(raw)
    output.with_suffix(".json").write_text(json.dumps(chosen,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--q8",action="store_true")
    p.add_argument("--reference",type=Path,
                   help="instead select every deployed Q8 reference matrix by exact name/shape")
    args=p.parse_args()
    extract(args.source,args.output,args.q8,args.reference)
    print(args.output,args.output.stat().st_size)


if __name__=="__main__":
    main()
