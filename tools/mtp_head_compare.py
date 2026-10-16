#!/usr/bin/env python3
"""Synthetic native-head requantization check; not a real-text quality gate."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-kl", type=float, default=0.005)
    args = parser.parse_args()
    reference = np.fromfile(args.reference, "<f4")
    candidate = np.fromfile(args.candidate, "<f4")
    if (reference.size != 25600 + 248320 or candidate.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()
            or not np.array_equal(reference[:12800], candidate[:12800])):
        raise ValueError("native requantization fixtures must have identical finite inputs and full logits")
    p = reference[25600:].astype(np.float64)
    q = candidate[25600:].astype(np.float64)
    p -= p.max()
    q -= q.max()
    log_p = p - np.log(np.exp(p).sum())
    log_q = q - np.log(np.exp(q).sum())
    kl = float(np.sum(np.exp(log_p) * (log_p - log_q)))
    ref_token, candidate_token = int(np.argmax(p)), int(np.argmax(q))
    result = dict(kind="native_mtp_synthetic_requant", synthetic_hc=True,
                  production_quality_qualified=False, acceptance_measured=False,
                  fixtures=1, reference_token=ref_token, candidate_token=candidate_token,
                  top1_equal=ref_token == candidate_token, kl_reference_candidate=kl,
                  max_kl=args.max_kl,
                  synthetic_gate_passed=kl <= args.max_kl and ref_token == candidate_token)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["synthetic_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
