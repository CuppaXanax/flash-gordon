#!/usr/bin/env python3
"""Quantize dense-only GGUF with llama.cpp's exact ggml_quantize_chunk API."""
import argparse
import array
import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from q4_source import header, string, SIZES


def target(t, protected, bf16_routers=False, mtp=False, mtp_experts_only=False):
    name, shape = t["name"], t["shape"]
    if mtp:
        if not name.startswith("mtp."):
            raise ValueError(f"non-MTP tensor in native overlay: {name}")
        if len(shape) < 2:
            return 0
        if mtp_experts_only and ".mlp.experts." not in name:
            return 8
        if ("block_inject_weight" in name or name.endswith(".mlp.gate.weight")
                or name.endswith(".shared_expert_gate.weight")):
            return 8
        if ".experts.down_proj" in name:
            return 7  # Q5_1 supports the native 640-wide rows.
        return 12 if shape[0] % 256 == 0 else 20
    if name == "token_embd.weight":
        return 8
    if name == "output.weight":
        return 14
    if bf16_routers and name.endswith(".ffn_gate_inp.weight"):
        return 30
    if "indexer." in name and name.endswith("_proj.weight"):
        return 30
    if (len(shape) < 2 or not name.endswith(".weight")
            or any(x in name for x in ("inject.", "ssm_alpha.", "ssm_beta.",
                                       "conv1d.", "ffn_gate_inp."))):
        return 0
    if shape[0] % 256:
        if shape[0] % 32:
            raise ValueError(f"unquantizable row: {name}")
        return 20
    if protected and re.search(r"\.(attn_qkv|ssm_out|attn_[qkv]|attn_output)\.weight$", name):
        return 13
    return 12


def floats(raw, kind):
    if sys.byteorder != "little":
        raise ValueError("quantizer requires a little-endian host")
    if kind == 0:
        return raw
    if kind == 30:
        shorts = array.array("H")
        shorts.frombytes(raw)
        return array.array("I", (x << 16 for x in shorts)).tobytes()
    if kind == 1:
        return array.array("f", (x[0] for x in struct.iter_unpack("<e", raw))).tobytes()
    raise ValueError(f"input is already quantized: {kind}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ggml-library", type=Path, required=True)
    p.add_argument("--protected", action="store_true")
    p.add_argument("--mtp", action="store_true",
                   help="experimental native MTP: Q4_K/IQ4_NL, Q5_1 expert down; preserve Q8 routers/injection")
    p.add_argument("--mtp-experts-only", action="store_true",
                   help="with --mtp, preserve every common projection and quantize only native routed experts")
    dense_precision = p.add_mutually_exclusive_group()
    dense_precision.add_argument("--q5-dense", action="store_true",
                   help="use Q5_K for all Q4_K-eligible matrices; non-256 rows remain IQ4_NL")
    dense_precision.add_argument("--q6-dense", action="store_true",
                   help="use Q6_K for all Q4_K/Q5_K-eligible matrices; non-256 rows remain IQ4_NL")
    p.add_argument("--q8-control", action="store_true",
                   help="requantize the same dense projection set to Q8_0")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--imatrix", type=Path)
    p.add_argument("--q8-overrides", type=Path,
                   help="deployed Q8 GGUF for explicitly identified source mismatches")
    p.add_argument("--override-map", type=Path,
                   help="q4_compare_source JSON; only its differing tensors are overridden")
    p.add_argument("--deployed-dense-only", action="store_true",
                   help="emit only matrices named by --q8-overrides; preserve head/embedding/router in the pack")
    p.add_argument("--bf16-routers", action="store_true", default=True, help="default: narrow only if every stored F32 value is BF16-exact")
    p.add_argument("--keep-f32-routers", action="store_false", dest="bf16_routers")
    args = p.parse_args()
    if args.mtp_experts_only and not args.mtp:
        p.error("--mtp-experts-only requires --mtp")
    if args.mtp and (args.q6_dense or args.protected or args.bf16_routers is False
                     or args.q8_overrides or args.deployed_dense_only):
        p.error("--mtp is a separate native-overlay mode; use --q5-dense for protected native matrices")
    if bool(args.q8_overrides) != bool(args.override_map):
        p.error("--q8-overrides and --override-map are required together")
    if args.deployed_dense_only and not args.q8_overrides:
        p.error("--deployed-dense-only requires the deployed Q8 reference")
    if args.threads < 1:
        p.error("--threads must be positive")
    lib = ctypes.CDLL(str(args.ggml_library.resolve()))
    row_size = lib.ggml_row_size
    row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    row_size.restype = ctypes.c_size_t
    for kind, (block, size) in SIZES.items():
        if row_size(kind, 256) != 256 // block * size:
            raise ValueError(f"GGML ABI block-size mismatch: type {kind}")
    if sys.byteorder != "little":
        raise ValueError("quantizer requires a little-endian host")
    converters = {1: lib.ggml_fp16_to_fp32_row, 30: lib.ggml_bf16_to_fp32_row,
                  8: lib.dequantize_row_q8_0}
    for convert in converters.values():
        convert.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        convert.restype = None
    quant = lib.ggml_quantize_chunk
    quant.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    quant.restype = ctypes.c_size_t
    metadata, ts, source_base, align = header(str(args.source))
    overrides, override_base = {}, 0
    if args.q8_overrides:
        _, reference, override_base, _ = header(str(args.q8_overrides))
        reference = {t["name"]: t for t in reference}
        mapping = json.loads(args.override_map.read_text())
        with args.q8_overrides.open("rb") as file:
            for row in mapping["rows"]:
                if row["identical"]:
                    continue
                name = row["name"]
                if name in overrides:
                    raise ValueError(f"duplicate source override: {name}")
                t = reference.get(name)
                if not t or t["type"] != 8 or t["shape"] != row["shape"]:
                    raise ValueError(f"source override type/shape: {name}")
                n = math.prod(t["shape"])//32*34
                file.seek(override_base+t["offset"])
                payload = file.read(n)
                if len(payload) != n or hashlib.sha256(payload).hexdigest() != row["production_sha256"]:
                    raise ValueError(f"source override hash: {name}")
                overrides[name] = dict(t, sha256=row["production_sha256"])
        if not overrides or set(overrides)-{t["name"] for t in ts}:
            raise ValueError("empty or unmatched source overrides")
        if args.deployed_dense_only:
            if any(t["type"] != 8 or len(t["shape"]) != 2 or t["name"] in
                   ("output.weight","token_embd.weight") for t in reference.values()):
                raise ValueError("deployed dense reference must contain only nonhead Q8 matrices")
            ts = [t for t in ts if t["name"] in reference]
            if {t["name"] for t in ts} != set(reference):
                raise ValueError("missing deployed dense source tensors")
    importance = {}
    imatrix_report = None
    if args.imatrix and not args.q8_control:
        _, entries, im_base, _ = header(str(args.imatrix))
        entries = {t["name"]: t for t in entries}
        with args.imatrix.open("rb") as file:
            for t in ts:
                if target(t, args.protected, mtp=args.mtp, mtp_experts_only=args.mtp_experts_only) not in (12, 13, 14, 20):
                    continue
                sums = entries.get(t["name"]+".in_sum2")
                counts = entries.get(t["name"]+".counts")
                if not sums or not counts:
                    continue
                if (sums["type"] != 0 or sums["shape"] != [t["shape"][0]]
                        or counts["type"] != 0 or counts["shape"] != [1]):
                    raise ValueError(f"imatrix shape/type mismatch: {t['name']}")
                file.seek(im_base+counts["offset"])
                count, = struct.unpack("<f", file.read(4))
                if not math.isfinite(count) or count < 1 or count != round(count):
                    raise ValueError(f"invalid imatrix count: {t['name']}")
                file.seek(im_base+sums["offset"])
                values = array.array("f")
                values.frombytes(file.read(t["shape"][0]*4))
                if len(values) != t["shape"][0] or any(
                        not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError(f"invalid imatrix sums: {t['name']}")
                normalized = array.array("f", (v/count for v in values))
                if not any(normalized):
                    raise ValueError(f"zero imatrix: {t['name']}")
                importance[t["name"]] = ctypes.create_string_buffer(normalized.tobytes())
        imatrix_report = dict(path=str(args.imatrix),
            sha256=hashlib.sha256(args.imatrix.read_bytes()).hexdigest(),
            mapped=len(importance),
            missing=[t["name"] for t in ts
                     if target(t, args.protected, mtp=args.mtp, mtp_experts_only=args.mtp_experts_only) in (12, 13, 14, 20)
                     and t["name"] not in importance])
    keys = (string("general.file_type"), string("general.quantization_version"))
    metadata = [item for item in metadata if not item.startswith(keys)]
    metadata.extend((keys[0]+struct.pack("<II", 4, 7 if args.q8_control else 18 if args.q6_dense else 17 if args.protected or args.q5_dense else 15),
                     keys[1]+struct.pack("<II", 4, 2)))
    cursor = 0
    for t in ts:
        t["source_type"], t["source_offset"] = t["type"], t["offset"]
        if t["name"] in overrides:
            ref = overrides[t["name"]]
            if t["shape"] != ref["shape"]:
                raise ValueError(f"BF16/reference shape mismatch: {t['name']}")
            t["bf16_source_type"] = t["source_type"]
            t["source_type"], t["source_offset"] = 8, ref["offset"]
            t["source_reference_sha256"] = ref["sha256"]
        t["type"] = target(t, args.protected, args.bf16_routers, args.mtp, args.mtp_experts_only)
        if args.q5_dense and t["type"] == 12:
            t["type"] = 13
        if args.q6_dense and t["type"] in (12, 13):
            t["type"] = 14
        if args.q8_control and (t["type"] in (12, 13, 14, 20) or args.mtp and t["type"] == 7):
            t["type"] = 8
        if t["type"] == 30 and t["source_type"] == 0:
            with args.source.open("rb") as file:
                file.seek(source_base+t["source_offset"])
                raw = file.read(t["shape"][0]*t["shape"][1]*4)
            words = array.array("I")
            words.frombytes(raw)
            if any(w & 65535 for w in words):
                t["type"] = 0
            t["lossless_bf16"] = t["type"] == 30
        count = 1
        for dim in t["shape"]:
            count *= dim
        block, bs = SIZES[t["type"]]
        t["bytes"] = count // block * bs
        cursor = (cursor + align - 1)//align*align
        t["offset"] = cursor
        cursor += t["bytes"]
    h = b"GGUF" + struct.pack("<IQQ", 3, len(ts), len(metadata)) + b"".join(metadata)
    for t in ts:
        h += (string(t["name"]) + struct.pack("<I", len(t["shape"]))
              + struct.pack("<"+"Q"*len(t["shape"]), *t["shape"])
              + struct.pack("<IQ", t["type"], t["offset"]))
    h += bytes((-len(h)) % align)
    partial = args.output.with_suffix(".partial")
    def encode(raw, n, width, source_type, kind, weights):
        if args.mtp and source_type == kind:
            return raw
        if source_type in converters:
            converted = ctypes.create_string_buffer(n*width*4)
            converters[source_type](ctypes.create_string_buffer(raw), converted, n*width)
            f32 = converted.raw
        else:
            f32 = floats(raw, source_type)
        if kind == 0:
            return f32
        if kind == 30:
            if source_type == 30:
                return raw
            words = array.array("I")
            words.frombytes(f32)
            if any(w & 65535 for w in words):
                raise ValueError("BF16 narrowing lost bits")
            return array.array("H", (w >> 16 for w in words)).tobytes()
        block, bs = SIZES[kind]
        expected = n*width//block*bs
        output = ctypes.create_string_buffer(expected)
        inp = ctypes.create_string_buffer(f32)
        size = quant(kind, inp, output, 0, n, width, weights)
        if size != expected:
            raise ValueError("quantizer byte mismatch")
        return output.raw

    with args.source.open("rb") as src, \
            (args.q8_overrides or args.source).open("rb") as reference, partial.open("wb") as dst, \
            ThreadPoolExecutor(max_workers=args.threads) as pool:
        dst.write(h)
        for t in ts:
            input_file = reference if t["name"] in overrides else src
            input_base = override_base if t["name"] in overrides else source_base
            input_file.seek(input_base+t["source_offset"])
            dst.seek(len(h)+t["offset"])
            width = t["shape"][0]
            rows = 1
            for dim in t["shape"][1:]:
                rows *= dim
            digest = hashlib.sha256()
            produced = 0
            pending = deque()
            def write_result():
                nonlocal produced
                result = pending.popleft().result()
                dst.write(result)
                digest.update(result)
                produced += len(result)
            for first in range(0, rows, 64):
                n = min(64, rows-first)
                source_block, source_size = SIZES[t["source_type"]]
                source_bytes = n*width//source_block*source_size
                raw = input_file.read(source_bytes)
                if len(raw) != source_bytes:
                    raise ValueError(f"short source tensor: {t['name']}")
                pending.append(pool.submit(encode, raw, n, width,
                                           t["source_type"], t["type"],
                                           importance.get(t["name"])))
                if len(pending) >= args.threads*2:
                    write_result()
            while pending:
                write_result()
            if produced != t["bytes"]:
                raise ValueError(f"tensor byte mismatch: {t['name']}")
            t["sha256"] = digest.hexdigest()
            print(t["name"], t["type"], t["bytes"], flush=True)
    partial.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        source=str(args.source), protected=args.protected, q5_dense=args.q5_dense,
        q6_dense=args.q6_dense, mtp=args.mtp, mtp_experts_only=args.mtp_experts_only,
        q8_control=args.q8_control, threads=args.threads,
        imatrix=imatrix_report,
        q8_source_overrides=str(args.q8_overrides) if args.q8_overrides else None,
        q8_source_override_count=len(overrides),
        router=("unchanged native source Q8_0" if args.mtp else
                "BF16 only when F32 round-trip is exact; otherwise F32"), tensors=ts), indent=2)+"\n")
    lib.ggml_quantize_free()


if __name__ == "__main__":
    main()
