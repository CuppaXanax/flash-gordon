#!/usr/bin/env python3
"""Compare deployed, uncooked Q8 tensors against the BF16 pipeline Q8 control."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from q4_source import header, SIZES


def compare(production, control):
    _, old, old_base, _ = header(str(production))
    _, new, new_base, _ = header(str(control))
    new = {t["name"]: t for t in new}
    results = []
    with production.open("rb") as a, control.open("rb") as b:
        for t in old:
            r = new.get(t["name"])
            if not r or r["shape"] != t["shape"] or r["type"] != 8 or t["type"] != 8:
                raise ValueError(f"missing/type/shape mismatch: {t['name']}")
            block, size = SIZES[t["type"]]
            count = math.prod(t["shape"])//block*size
            a.seek(old_base+t["offset"]);b.seek(new_base+r["offset"])
            left, right = a.read(count), b.read(count)
            if len(left) != count or len(right) != count:
                raise ValueError("short Q8 payload")
            item = dict(name=t["name"], shape=t["shape"], bytes=count,
                        production_sha256=hashlib.sha256(left).hexdigest(),
                        control_sha256=hashlib.sha256(right).hexdigest(),
                        identical=left == right)
            if left != right:
                ratios, code_equal, dot, aa, bb, error = [], 0, 0., 0., 0., 0.
                samples = sorted(set(i*(count//34-1)//127 for i in range(128)))
                for index in samples:
                    x, y = left[index*34:(index+1)*34], right[index*34:(index+1)*34]
                    dx, dy = struct.unpack("<e", x[:2])[0], struct.unpack("<e", y[:2])[0]
                    if dy: ratios.append(dx/dy)
                    code_equal += x[2:] == y[2:]
                    for qx, qy in zip(struct.unpack("32b",x[2:]),struct.unpack("32b",y[2:])):
                        vx, vy = dx*qx, dy*qy
                        dot += vx*vy;aa += vx*vx;bb += vy*vy;error += (vx-vy)**2
                item.update(sampled_blocks=len(samples), identical_code_blocks=code_equal,
                            scale_ratio_min=min(ratios) if ratios else None,
                            scale_ratio_median=statistics.median(ratios) if ratios else None,
                            scale_ratio_max=max(ratios) if ratios else None,
                            cosine=dot/math.sqrt(aa*bb) if aa*bb else None,
                            relative_l2=math.sqrt(error/aa) if aa else None)
            results.append(item)
    return dict(tensors=len(results), identical=sum(t["identical"] for t in results),
                different=sum(not t["identical"] for t in results), rows=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production",type=Path,required=True)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--require-identical",action="store_true",
                   help="fail the provenance gate if any deployed tensor differs")
    args = p.parse_args()
    result = compare(args.production,args.control)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}))
    for row in result["rows"]:
        if not row["identical"]:
            print(json.dumps(row))
    if args.require_identical and result["different"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
