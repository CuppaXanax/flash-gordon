#!/usr/bin/env python3
"""Direct full-vocabulary comparisons with explicit corpus and reference binding."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def logits(path):
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:8] != b"FGLOG01\n" or struct.unpack_from("<I", raw, 8)[0] != 248320:
        raise ValueError(f"invalid full-vocabulary header: {path}")
    if len(raw) != 12 + 248320 * 4:
        raise ValueError(f"invalid full-vocabulary length: {path}")
    values = np.frombuffer(raw, dtype="<f4", offset=12).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite logits: {path}")
    return values, hashlib.sha256(raw).hexdigest()


def compare(reference, candidate, corpus):
    items = json.loads(corpus.read_text(encoding="utf-8-sig"))
    for root in (reference, candidate):
        if json.loads((root / "corpus.json").read_text(encoding="utf-8-sig")) != items:
            raise ValueError(f"corpus mismatch: {root}")
    rows = []
    for key, text in items:
        a, ah = logits(reference / (key + ".fgl"))
        b, bh = logits(candidate / (key + ".fgl"))
        la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
        lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
        rows.append(dict(id=key, kl=max(0., float(np.sum(np.exp(la) * (la - lb)))),
                         top1=bool(a.argmax() == b.argmax()),
                         reference_sha256=ah, candidate_sha256=bh,
                         text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    values = np.array([r["kl"] for r in rows])
    agreement = sum(r["top1"] for r in rows) / len(rows)
    return dict(reference=str(reference), candidate=str(candidate),
                corpus_file_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                samples=len(rows), kl_mean=float(values.mean()), kl_max=float(values.max()),
                kl_p99=float(np.sort(values)[int(np.ceil(.99 * len(rows))) - 1]),
                top1_agreement=agreement,
                strict_gate=bool(values.mean() <= .005 and agreement >= .985), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.reference, a.candidate, a.corpus)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
