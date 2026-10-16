#!/usr/bin/env python3
"""Check every dumped block output and recurrent-state element."""
import argparse
from array import array
import json
import math
from pathlib import Path


def compare(reference, candidate, tolerance, relative):
    left_bytes, right_bytes = Path(reference).read_bytes(), Path(candidate).read_bytes()
    if len(left_bytes) != len(right_bytes) or not left_bytes or len(left_bytes) % 4:
        raise ValueError("incompatible or empty float dumps")
    left, right = array("f"), array("f")
    left.frombytes(left_bytes)
    right.frombytes(right_bytes)
    maximum = scaled = 0.0
    changed = 0
    for a, b in zip(left, right):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("nonfinite dump")
        error = abs(a - b)
        maximum = max(maximum, error)
        scaled = max(scaled, error / max(1.0, abs(a)) if relative else error)
        changed += a != b
    if scaled > tolerance:
        raise ValueError(f"dump error {scaled} exceeds {tolerance}")
    return {"elements": len(left), "changed": changed, "bit_exact": left_bytes == right_bytes,
            "max_abs": maximum, "max_scaled": scaled, "tolerance": tolerance, "relative": relative}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    print(json.dumps({
        "output": compare(args.reference, args.candidate, 2e-6, True),
        "state": compare(args.reference + ".gdn", args.candidate + ".gdn", 3e-5, False),
    }))
