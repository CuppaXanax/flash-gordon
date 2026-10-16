#!/usr/bin/env python3
"""Fail-closed inventory of target/native formats before a serving lease."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

TYPES = {0: "F32", 7: "Q5_1", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K",
         20: "IQ4_NL", 30: "BF16"}
LAYOUTS = {0: "GGML", 1: "Q8_COOKED", 2: "K_EXPERT_COOKED",
           3: "Q5_1_EXPERT_COOKED", 4: "HOST_Q8", 5: "Q8_EXPERT_COOKED",
           6: "DENSE_SOA", 7: "DENSE_COMPACT"}
PINNED_MANIFEST = "0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252"


def load_manifest(path):
    raw = Path(path).read_bytes()
    if len(raw) < 184 or raw[:8] != b"QOWNFGM1":
        raise ValueError("invalid manifest header")
    version, protocol, size, ranks, layers, experts = struct.unpack_from("<6I", raw, 8)
    # The file is the C ABI, including fixed-size arrays, not a packed wire format.
    if (version, protocol, ranks, layers, experts) != (6, 6, 8, 48, 512):
        raise ValueError("unsupported manifest ABI; update alongside fg_manifest.h")
    if size != len(raw):
        raise ValueError("manifest header/file size mismatch")
    count = struct.unpack_from("<I", raw, 60)[0]
    if not 0 < count <= 4096:
        raise ValueError("invalid tensor count")
    start = 184 + ranks * 120 + layers + layers * 4 + layers * experts * 2
    start = (start + 7) & ~7
    records = []
    for index in range(count):
        offset = start + index * 200
        name_bytes = raw[offset:offset + 96]
        if b"\0" not in name_bytes:
            raise ValueError("unterminated tensor name")
        name = name_bytes.split(b"\0")[0].decode("utf-8")
        position, size, kind, dims = struct.unpack_from("<QQII", raw, offset + 96)
        shape = list(struct.unpack_from("<4Q", raw, offset + 120))
        rank, layer, expert, family_kind, layout = struct.unpack_from("<HHHBB", raw, offset + 152)
        if not name or not 0 < dims <= 4 or not all(shape[:dims]):
            raise ValueError(f"invalid tensor record {index}")
        records.append(dict(name=name, offset=position, bytes=size, type=kind,
                            dims=dims, shape=shape[:dims], rank=rank, layer=layer,
                            expert=expert, kind=family_kind, layout=layout,
                            sha256=raw[offset + 164:offset + 196].hex()))
    if len({record["name"] for record in records}) != count:
        raise ValueError("duplicate manifest tensor names")
    return dict(sha256=hashlib.sha256(raw).hexdigest(), records=records,
                rank_count=ranks, layer_count=layers, expert_count=experts,
                tensor_count=count, bytes=len(raw))


def gguf_header(raw):
    cursor = 0

    def take(size):
        nonlocal cursor
        if size < 0 or cursor + size > len(raw):
            raise ValueError("GGUF header exceeds supplied bytes")
        result = raw[cursor:cursor + size]
        cursor += size
        return result

    def unpack(fmt):
        return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

    def string():
        size = unpack("Q")
        return take(size).decode("utf-8")

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = unpack("I"), unpack("Q")
            if count > 100_000_000:
                raise ValueError("unbounded GGUF array")
            if subtype not in (8, 9):
                widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                          7: 1, 10: 8, 11: 8, 12: 8}
                take(widths[subtype] * count)
            else:
                for _ in range(count):
                    value(subtype)
            return None
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
        return unpack(formats[kind])

    if take(4) != b"GGUF" or unpack("I") != 3:
        raise ValueError("expected GGUF v3")
    tensors, pairs = unpack("Q"), unpack("Q")
    if tensors > 100_000 or pairs > 100_000:
        raise ValueError("unbounded GGUF header")
    metadata = {}
    for _ in range(pairs):
        key, kind = string(), unpack("I")
        data = value(kind)
        if key in ("general.alignment", "general.architecture", "flash_gordon.mtp.source_revision"):
            metadata[key] = data
    records = []
    widths = {0: (1, 4), 1: (1, 2), 7: (32, 24), 8: (32, 34),
              12: (256, 144), 13: (256, 176), 20: (32, 18), 30: (1, 2)}
    for _ in range(tensors):
        name, dims = string(), unpack("I")
        if not 0 < dims <= 4:
            raise ValueError("invalid GGUF dimensions")
        shape = [unpack("Q") for _ in range(dims)]
        kind, offset = unpack("I"), unpack("Q")
        block, width = widths[kind]
        count = math.prod(shape)
        if not count or count % block:
            raise ValueError("invalid GGUF block geometry")
        records.append(dict(name=name, shape=shape, type=kind, offset=offset,
                            bytes=count // block * width))
    alignment = metadata.get("general.alignment", 32)
    if not alignment or alignment & (alignment - 1):
        raise ValueError("invalid GGUF alignment")
    start = (cursor + alignment - 1) & ~(alignment - 1)
    for record in records:
        record["file_offset"] = start + record["offset"]
    return dict(metadata=metadata, records=records, data_offset=start,
                header_sha256=hashlib.sha256(raw[:start]).hexdigest())


def target_support(record):
    family = re.sub(r"^blk\.\d+\.", "", re.sub(r"\.rank\d+$", "", record["name"]))
    pair = record["type"], record["layout"]
    route = None
    expected = set()
    if family in {"attn_gate.weight", "attn_qkv.weight", "ple_key.weight", "ple_value.weight"}:
        expected, route = {(12, 7)}, "fg_vk_dense_quant: compact Q4_K small-N"
    elif family in {"ssm_out.weight", "hc_attn_down.weight", "hc_ffn_down.weight",
                    "output_hc_down.weight"}:
        expected, route = {(13, 7)}, "fg_vk_dense_quant: compact Q5_K small-N; output HC uses N1"
    elif family in {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"}:
        expected, route = {(12, 6)}, "owner_shared_expert -> fg_vk_dense_quant: SOA Q4_K"
    elif family == "ffn_down_shexp.weight":
        expected, route = {(20, 7)}, "owner_shared_expert -> fg_vk_dense_quant: compact IQ4_NL"
    elif family in {"attn_q.weight", "attn_output.weight", "hc_attn_up.weight",
                    "hc_ffn_up.weight", "output_hc_up.weight", "output.weight"}:
        expected, route = {(8, 1)}, "fg_vk_dense_q8_0_cooked; output/head use shared native N1"
    elif family in {"attn_k.weight", "attn_v.weight"}:
        expected, route = {(8, 0)}, "model runtime_layout cooks Q8; fg_vk_dense_q8_0_cooked"
    elif family == "token_embd.weight":
        expected, route = {(8, 0)}, "target embedding GPU Q8; native uses exact exported sidecar"
    elif family in {"ffn_gate_exps.weight", "ffn_up_exps.weight"}:
        expected, route = {(12, 2), (13, 2)}, "fg_vk_small_n_experts: cooked K gate/up"
    elif family == "ffn_down_exps.weight":
        expected, route = {(7, 3), (8, 5)}, "fg_vk_small_n_experts: cooked Q5_1 / exact native-lane Q8 down"
    elif family == "ffn_gate_inp.weight":
        expected, route = {(30, 0)}, "fg_vk_dense_saved_format -> BF16_ROUTER -> small_plain_dispatch"
    elif family in {"indexer.k_proj.weight", "indexer.q_proj.weight"}:
        expected, route = {(30, 0)}, "QSA -> fg_vk_dense_bf16_f32, per-token causal index selection"
    elif family == "per_layer_token_embd.weight":
        expected, route = {(20, 0)}, "ngram CPU/IQ4_NL row decode; PLE consumes F32 rows sequentially"
    elif family == "tokenizer/tokenizer.fgt":
        expected, route = {(0, 0)}, "tokenizer artifact, not a weight or a GPU dispatch"
    elif family in {
            "attn_k_norm.weight", "attn_q_norm.weight", "ffn_gate_inp_shexp.weight",
            "hc_attn_inject.weight", "hc_attn_norm.weight", "hc_ffn_inject.weight",
            "hc_ffn_norm.weight", "indexer.k_norm.weight", "indexer.q_norm.weight",
            "output_hc_norm.weight", "ple_conv1d.weight", "ple_norm_conv.weight",
            "ple_norm_key.weight", "ple_norm_query.weight", "ssm_a", "ssm_alpha.weight",
            "ssm_beta.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight"}:
        expected, route = {(0, 0)}, "F32 RMS/HC injection, control dense, convolution and sequential recurrence"
    if pair not in expected:
        raise ValueError(f"UNSUPPORTED {record['name']}: type/layout={pair}")
    if record["kind"] == 2 and (record["dims"] != 3 or record["shape"][2] != 512):
        raise ValueError("small-N requires full owned expert slabs")
    return family, route


def audit(manifest_path, overlay_path, expected_sha=PINNED_MANIFEST):
    manifest = load_manifest(manifest_path)
    if manifest["sha256"] != expected_sha:
        raise ValueError("manifest differs from the requested sealed production inventory")
    repo = Path(__file__).resolve().parent.parent
    vk_source = (repo / "src" / "vk.c").read_text()
    if not re.search(r"fg_status\s+fg_vk_small_n_q8_down\s*\(", vk_source):
        raise ValueError("source lacks the exact cooked-Q8 small-N down implementation")
    groups = {}
    for record in manifest["records"]:
        family, route = target_support(record)
        key = family, record["type"], record["layout"]
        group = groups.setdefault(key, dict(family=family, format=TYPES[record["type"]],
                                           layout=LAYOUTS[record["layout"]], route=route,
                                           count=0, ranks=set(), layers=set(), shapes=set()))
        group["count"] += 1
        group["ranks"].add(record["rank"])
        group["layers"].add(record["layer"])
        group["shapes"].add(tuple(record["shape"]))
    target = []
    for _, group in sorted(groups.items()):
        for key in ("ranks", "layers", "shapes"):
            group[key] = sorted(group[key])
        target.append(group)
    with Path(overlay_path).open("rb") as file:
        native = gguf_header(file.read(16 * 1024 * 1024))
        if any(record["file_offset"] + record["bytes"] > file.seek(0, 2)
               for record in native["records"]):
            raise ValueError("truncated native payload")
        file.seek(0)
        native_sha = hashlib.file_digest(file, "sha256").hexdigest()
    source = (repo / "src" / "mtp_head.c").read_text()
    aliases = re.findall(
        r'\{"([^"]+)","([^"]+)",(\d+),(\d+),(\d+),(true|false),(true|false),(\d+)\}', source)
    if not aliases:
        raise ValueError("missing native importer alias inventory")
    used, mappings = set(), []
    for name, alias, inputs, rows, experts, f32, norm, half in aliases:
        matches = [record for record in native["records"] if record["name"] in (name, "mtp." + name)]
        if len(matches) != 1:
            raise ValueError(f"missing/duplicate native alias {name}")
        record = matches[0]
        if record["type"] != (0 if norm == "true" else 8):
            raise ValueError(f"native overlay needs an explicit new format audit: {record['name']}")
        shape = [int(inputs)] if norm == "true" else [int(inputs), int(rows)]
        if int(experts) > 1:
            shape = [int(inputs), int(rows) * (2 if "gate_up_proj" in name else 1), int(experts)]
        if record["shape"] != shape:
            raise ValueError(f"native shape mismatch {record['name']}")
        used.add(record["name"])
        route = ("F32 vector; norm gamma +1 once" if norm == "true" else
                 "Q8 -> F32 importer for router/control" if f32 == "true" else
                 "cooked Q8 experts; native N1 inline gate/up/down" if int(experts) > 1 else
                 "cooked Q8 dense; native N1 GPU")
        mappings.append(dict(family=record["name"], alias=alias, format=TYPES[record["type"]],
                             shape=record["shape"], runtime=route, half=int(half)))
    unused_indexer = {
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": (8, [2560, 640]),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": (0, [128]),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (0, [128]),
    }
    unused = []
    for record in native["records"]:
        if record["name"] not in unused_indexer:
            continue
        if (record["type"], record["shape"]) != unused_indexer[record["name"]]:
            raise ValueError("unexpected unused native indexer format/shape")
        used.add(record["name"])
        unused.append(dict(family=record["name"], format=TYPES[record["type"]],
                           shape=record["shape"],
                           runtime="Not dispatched: native drafts use full/bounded KV attention; "
                                   "target verification retains its learned QSA indexer."))
    if used != {record["name"] for record in native["records"]}:
        raise ValueError("unaccounted native overlay tensors")
    return dict(schema=1, manifest_sha256=manifest["sha256"],
                tensor_count=manifest["tensor_count"], target_family_formats=target,
                native_tensors=len(native["records"]), native_aliases=mappings,
                native_not_dispatched=unused,
                native_header_sha256=native["header_sha256"],
                native_overlay_sha256=native_sha,
                engine_sources={name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                                for name in ("src/vk.c", "src/model.c", "src/owner.c",
                                             "src/expert.c", "src/qsa.c", "src/mtp_head.c",
                                             "src/mtp_embedding.c", "src/output.c", "src/ngram.c")},
                supported=True, caveat="Format/shape inventory and dispatch audit, not full-model greedy qualification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.overlay)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {result['tensor_count']} target records / "
          f"{len(result['target_family_formats'])} family/formats; "
          f"{result['native_tensors']} native tensors / {len(result['native_aliases'])} aliases")


if __name__ == "__main__":
    main()
