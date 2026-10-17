#!/usr/bin/env python3
"""Byte-weight actual target-verifier routes, never sliding teacher routes."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from mtp_format_audit import load_manifest


def geometry(manifest):
    banks = defaultdict(dict)
    for record in manifest["records"]:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\.rank\d+", record["name"])
        if not match:
            continue
        layer, family = int(match[1]), match[2]
        shape, size = record["shape"], record["bytes"]
        if record["dims"] != 3 or shape[2] != 512 or size <= 0 or size % 512:
            raise ValueError("union analysis requires complete, uniform-stride owner expert banks")
        if family in banks[layer]:
            raise ValueError("duplicate expert bank")
        banks[layer][family] = dict(stride=size // 512, type=record["type"])
    for bank in banks.values():
        if set(bank) != {"gate", "up", "down"}:
            raise ValueError("incomplete expert geometry")
    return banks


def summarize(rows, manifest, provenance, layers=None):
    if provenance.get("origin") != "native-target-MTP-verification":
        raise ValueError("missing authentic target-verifier provenance")
    if provenance.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("capture/manifest mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", provenance.get("source_commit", "")):
        raise ValueError("missing source commit")
    sessions = provenance.get("sessions", {})
    excluded = set(provenance.get("excluded_sessions", []))
    expected = set(range(48) if layers is None else layers)
    if not expected or not expected <= set(range(48)):
        raise ValueError("invalid layer scope")
    banks = geometry(manifest)
    if not expected <= banks.keys():
        raise ValueError("manifest lacks requested expert banks")
    windows = defaultdict(set)
    totals = defaultdict(lambda: dict(n1_gu=0, n4_gu=0, n1_down=0, n4_down=0, observations=0))
    observations, ignored = [], defaultdict(int)
    for row in rows:
        if row.get("kind") != "mtp_verify_routes" or row.get("version") != 1:
            raise ValueError("not an append-only verification route record")
        session = str(row.get("session"))
        if session in excluded:
            ignored["excluded_session"] += 1
            continue
        if session not in sessions:
            raise ValueError(f"unbound session {session}")
        case = sessions[session]
        if case.get("category") not in ("chat", "code") or not case.get("case"):
            raise ValueError("invalid case/category binding")
        if type(row.get("replay")) is not bool:
            raise ValueError("missing replay classification")
        n, first, layer = row.get("n"), row.get("first"), row.get("layer")
        if any(type(x) is not int for x in (n, first, layer)) or not 1 <= n <= 4 or not 0 <= first <= 262144-n or layer not in range(48):
            raise ValueError("invalid verification shape")
        ids = row.get("experts")
        if not isinstance(ids, list) or len(ids) != n:
            raise ValueError("missing per-token expert selections")
        for token in ids:
            if not isinstance(token, list) or len(token) != 10 or any(type(i) is not int or not 0 <= i < 512 for i in token) or len(set(token)) != 10:
                raise ValueError("invalid top-10 selection")
        if row["replay"] or n != 4 or layer not in expected:
            ignored["replay" if row["replay"] else "other_shape_or_layer"] += 1
            continue
        key = (session, first)
        if layer in windows[key]:
            raise ValueError("duplicate target-verifier window/layer; do not join separate runs")
        windows[key].add(layer)
        union = set().union(*map(set, ids))
        gu = banks[layer]["gate"]["stride"] + banks[layer]["up"]["stride"]
        down = banks[layer]["down"]["stride"]
        value = dict(n1_gu=10*gu, n4_gu=len(union)*gu,
                     n1_down=10*down, n4_down=len(union)*down, observations=1)
        for group in ("all", case["category"], f"layer-{layer}"):
            for name, number in value.items():
                totals[group][name] += number
        observations.append(dict(session=session, first=first, layer=layer,
                                 case=case["case"], category=case["category"],
                                 unique_experts=len(union), **value))
    if not windows or any(owned != expected for owned in windows.values()):
        raise ValueError("empty or incomplete all-layer verification windows")
    for total in totals.values():
        total["gu_byte_ratio"] = total["n4_gu"] / total["n1_gu"]
        total["down_byte_ratio"] = total["n4_down"] / total["n1_down"]
    return dict(origin=provenance["origin"], manifest_sha256=manifest["sha256"],
                source_commit=provenance["source_commit"], layers=sorted(expected),
                full_model_scope=len(expected) == 48, windows=len(windows),
                denominator="pending token's ten experts within the same N4 target call",
                semantics="logical unique cooked-bank bytes, not measured DRAM counters or GU/down time",
                groups=dict(totals), ignored=dict(ignored), observations=observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", required=True, nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--layers", help="Explicit partial diagnostic scope, e.g. 42,43,44,45,46,47")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = []
    for root in args.routes:
        files.extend(sorted(root.rglob("verify-routes-*.jsonl")) if root.is_dir() else [root])
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
    layers = None if args.layers is None else [int(x) for x in args.layers.split(",")]
    result = summarize(rows, load_manifest(args.manifest),
                       json.loads(args.provenance.read_text()), layers)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
