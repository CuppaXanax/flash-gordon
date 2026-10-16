#!/usr/bin/env python3
"""Independent NumPy first-cell oracle for the Q8 native MTP overlay.

Uses raw fused HF-order GGUF tensors, not FG's cooked aliases. This is an
architecture/import check on synthetic HC, not real-text model qualification.
"""

import argparse
import json
import mmap
from pathlib import Path
import struct

import numpy as np


class Overlay:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        if self.data[:4] != b"GGUF":
            raise ValueError("not GGUF")
        self.pos = 4
        version, nt, nk = self.read("<IQQ")
        if version != 3:
            raise ValueError("unsupported GGUF version")
        for _ in range(nk):
            self.string()
            kind = self.read("<I")[0]
            if kind != 8:
                raise ValueError("unexpected overlay metadata")
            self.string()
        self.tensors = {}
        for _ in range(nt):
            name = self.string()
            dims = self.read("<I")[0]
            shape = self.read("<" + "Q" * dims)
            kind, offset = self.read("<IQ")
            self.tensors[name] = kind, tuple(reversed(shape)), offset
        self.base = (self.pos + 31) & ~31

    def read(self, fmt):
        values = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return values

    def string(self):
        n = self.read("<Q")[0]
        s = self.data[self.pos:self.pos + n].decode()
        self.pos += n
        return s

    def get(self, name, expert=None):
        kind, shape, offset = self.tensors["mtp." + name]
        if expert is not None:
            elements = int(np.prod(shape[1:]))
            offset += expert * elements // 32 * 34
            shape = shape[1:]
        count = int(np.prod(shape))
        if kind == 0:
            return np.frombuffer(self.data, "<f4", count, self.base + offset).reshape(shape).copy()
        if kind != 8:
            raise ValueError("expected Q8")
        blocks = np.frombuffer(self.data, np.uint8, count // 32 * 34,
                               self.base + offset).reshape(-1, 34)
        scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
        return (blocks[:, 2:].view(np.int8).astype(np.float32) * scales).reshape(shape)


def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def rms(x, gamma):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6)) * (gamma + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--head-q8")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = Overlay(a.overlay)
    dump = np.fromfile(a.dump, "<f4")
    hc, emb = dump[:10240], dump[10240:12800]
    actual_hc, actual_hidden = dump[12800:23040], dump[23040:25600]
    if len(actual_hidden) != 2560:
        raise ValueError("short native dump")
    w = source.get
    trace = {}
    trace_path = Path(a.dump + ".trace")
    if trace_path.exists():
        data = trace_path.read_bytes()
        offset = 0
        while offset < len(data):
            length, count = struct.unpack_from("<II", data, offset)
            offset += 8
            stage = data[offset:offset + length].decode()
            offset += length
            trace[stage] = np.frombuffer(data, "<f4", count, offset)
            offset += count * 4

    stage_errors = {}
    def compare_stage(name, expected):
        if name in trace:
            expected = np.asarray(expected).reshape(-1)
            difference = np.abs(trace[name] - expected)
            stage_errors[name] = float((difference / np.maximum(1, np.abs(expected))).max())
            print(f"{name}: max_scaled={stage_errors[name]:.9g}", flush=True)
            if not np.isfinite(trace[name]).all() or stage_errors[name] > 1e-4:
                raise AssertionError({"stage": name, "max_scaled": stage_errors[name]})

    def linear(name, x):
        return w(name) @ x

    def gr_read(x, prefix):
        n = rms(x.reshape(4, 2560), w(prefix + ".hc_norm.weight").reshape(4, 2560))
        lo = linear(prefix + ".input_mix_weight_down.weight", n.reshape(-1))
        lo = lo / 4
        up = linear(prefix + ".input_mix_weight_up.weight", lo * sigmoid(lo))
        mixed = (sigmoid(up.reshape(4, 2560)) * n).sum(axis=0) / 4
        inject = 2 * sigmoid(linear(prefix + ".block_inject_weight.weight", n.reshape(-1)) / 4)
        return mixed, inject

    en = linear("fc_embedding.weight", rms(emb, w("pre_fc_norm_embedding.weight")))
    hn = rms(hc, w("pre_fc_norm_hidden.weight")).reshape(4, 2560)
    state = hn @ w("fc_hidden.weight").T + en
    compare_stage("combined", state)
    mixed, inject = gr_read(state, "layers.0.attn_hyper_connection")
    compare_stage("attn_mixed", mixed)
    compare_stage("attn_injection", inject)
    qg = linear("layers.0.self_attn.q_proj.weight", mixed).reshape(24, 512)
    value = linear("layers.0.self_attn.v_proj.weight", mixed).reshape(2, 256)
    compare_stage("raw_q", qg)
    compare_stage("raw_v", value)
    attended = np.repeat(value, 12, axis=0) * sigmoid(qg[:, 256:])
    attn_out = linear("layers.0.self_attn.o_proj.weight", attended.reshape(-1))
    compare_stage("attention_output", attn_out)
    state = state + inject[:, None] * attn_out
    compare_stage("after_attention", state)
    mixed, inject = gr_read(state, "layers.0.mlp_hyper_connection")
    compare_stage("ffn_mixed", mixed)
    compare_stage("ffn_injection", inject)
    router = linear("layers.0.mlp.gate.weight", mixed)
    ids = np.argsort(-router, kind="stable")[:10]
    gates = np.exp(router[ids].astype(np.float64) - router.max())
    gates = (gates / gates.sum()).astype(np.float32)
    compare_stage("router_ids", ids)
    compare_stage("router_gates", gates)
    blocks = mixed.reshape(-1, 256)
    scale = np.max(np.abs(blocks), axis=1, keepdims=True) / np.float32(127)
    inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
    quantized = (np.rint(blocks * inverse) * scale).reshape(-1)
    compare_stage("quantized", quantized)
    experts = np.zeros(2560, np.float32)
    all_mid = []
    for expert, gate in zip(ids, gates):
        gu = w("layers.0.mlp.experts.gate_up_proj", int(expert)) @ quantized
        mid = gu[:640] * sigmoid(gu[:640]) * gu[640:]
        all_mid.append(mid)
        experts += gate * (w("layers.0.mlp.experts.down_proj", int(expert)) @ mid)
    compare_stage("expert_result_0", experts)
    compare_stage("expert_mid", np.concatenate(all_mid))
    g = linear("layers.0.mlp.shared_expert.gate_proj.weight", mixed)
    u = linear("layers.0.mlp.shared_expert.up_proj.weight", mixed)
    shared = linear("layers.0.mlp.shared_expert.down_proj.weight", g * sigmoid(g) * u)
    scalar = sigmoid(linear("layers.0.mlp.shared_expert_gate.weight", mixed)).item()
    compare_stage("moe_output", experts + scalar * shared)
    state = state + inject[:, None] * (experts + scalar * shared)
    compare_stage("final_hc", state)
    n = rms(state, w("hyper_connection_mixer.hc_norm.weight").reshape(4, 2560))
    lo = linear("hyper_connection_mixer.input_mix_weight_down.weight", n.reshape(-1))
    lo = lo / 4
    up = linear("hyper_connection_mixer.input_mix_weight_up.weight", lo * sigmoid(lo))
    hidden = (sigmoid(up.reshape(4, 2560)) * n).sum(axis=0) / 4
    result = dict(kind="native_head_numpy_reference", synthetic_hc=True,
                  expert_ids=ids.tolist(), model_quality_qualified=False,
                  stage_errors=stage_errors)
    for label, actual, expected in (("hc", actual_hc, state.reshape(-1)),
                                     ("contracted", actual_hidden, hidden)):
        difference = np.abs(actual - expected)
        result[label + "_max_abs"] = float(difference.max())
        result[label + "_max_scaled"] = float((difference / np.maximum(1, np.abs(expected))).max())
        if not np.isfinite(actual).all() or result[label + "_max_scaled"] > 1e-4:
            raise AssertionError(result)
    if a.head_q8:
        weights = np.memmap(a.head_q8, np.uint8, mode="r").reshape(248320, 80, 34)
        logits = []
        for first in range(0, 248320, 4096):
            block = np.asarray(weights[first:first + 4096])
            scales = block[:, :, :2].copy().view("<f2").astype(np.float32)
            matrix = (block[:, :, 2:].view(np.int8).astype(np.float32) * scales).reshape(-1, 2560)
            logits.append(matrix @ hidden)
        expected = np.concatenate(logits).astype(np.float64)
        actual = dump[25600:].astype(np.float64)
        if len(actual) != 248320 or not np.isfinite(actual).all():
            raise ValueError("missing/nonfinite GPU vocabulary dump")
        logp = expected - expected.max(); logp -= np.log(np.exp(logp).sum())
        p0 = np.exp(logp)
        logq = actual - actual.max(); logq -= np.log(np.exp(logq).sum())
        result["vocabulary_kl_reference_gpu"] = float(np.sum(p0 * (logp - logq)))
        result["greedy_equal"] = bool(expected.argmax() == actual.argmax())
        result["gpu_token"] = int(actual.argmax())
        if result["vocabulary_kl_reference_gpu"] > 0.005 or not result["greedy_equal"]:
            raise AssertionError(result)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
