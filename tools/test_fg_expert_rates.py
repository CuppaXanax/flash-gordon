#!/usr/bin/env python3
"""Unit tests for tools/fg_expert_rates.py (offline, no Vulkan)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fg_expert_rates  # noqa: E402

SAMPLE = "\n".join([
    "DECODE_PROFILE_KERNEL rank=0 scope=expert_decode kernel=fg_moe_decode_gate_up.spv calls=6 gpu_ms=0.607",
    "DECODE_PROFILE_KERNEL rank=0 scope=expert_decode kernel=fg_moe_decode_down_reduce.spv calls=6 gpu_ms=0.396",
    "DECODE_PROFILE_KERNEL rank=1 scope=expert_decode kernel=fg_moe_decode_gate_up.spv calls=6 gpu_ms=1.000",
    "DECODE_PROFILE_KERNEL rank=1 scope=expert_decode kernel=fg_moe_decode_down_reduce.spv calls=6 gpu_ms=1.000",
    "DECODE_PROFILE_KERNEL rank=0 scope=expert_activation kernel=fg_swiglu.spv calls=6 gpu_ms=0.005",
    "DECODE_PROFILE_KERNEL rank=0 scope=expert_reduce kernel=fg_moe_reduce.spv calls=6 gpu_ms=0.010",
    "DECODE_PROFILE_KERNEL rank=0 scope=gr_attn_read kernel=fg_group_rms_norm.spv calls=6 gpu_ms=0.087",
    "unrelated log line",
]) + "\n"


def write_log(directory: Path) -> Path:
    path = directory / "profile.log"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


class ExpertRatesTest(unittest.TestCase):
    def test_rows_and_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_log(Path(directory))
            capture = io.StringIO()
            with redirect_stdout(capture):
                status = fg_expert_rates.main([str(path), "--json"])
            self.assertEqual(status, 0)
            result = json.loads(capture.getvalue())
        rows = {(row["rank"], row["scope"], row["kernel"]): row for row in result["kernels"]}
        gate_up = rows[(0, "expert_decode", "fg_moe_decode_gate_up.spv")]
        self.assertEqual(gate_up["bytes"], 110_592_000)
        self.assertGreater(gate_up["achieved_gbps"], 180.0)
        self.assertLess(gate_up["achieved_gbps"], 185.0)
        down = rows[(0, "expert_decode", "fg_moe_decode_down_reduce.spv")]
        self.assertEqual(down["bytes"], 73_728_000)
        self.assertGreater(down["achieved_gbps"], 185.0)
        self.assertLess(down["achieved_gbps"], 190.0)
        unpriced = rows[(0, "gr_attn_read", "fg_group_rms_norm.spv")]
        self.assertFalse(unpriced["priced"])
        self.assertIsNone(unpriced["achieved_gbps"])

    def test_manifest_layer_totals(self):
        self.assertEqual(fg_expert_rates.RANK_LAYERS[1], list(range(0, 6)))
        self.assertEqual(fg_expert_rates.RANK_LAYERS[7], list(range(42, 48)))
        self.assertEqual(fg_expert_rates.RANK_GATE_UP[0], 110_592_000)
        self.assertEqual(fg_expert_rates.RANK_DOWN[0], 73_728_000)
        self.assertEqual(fg_expert_rates.RANK_GATE_UP[1], 114_688_000)
        self.assertEqual(fg_expert_rates.RANK_DOWN[1], 83_968_000)
        fleet = sum(fg_expert_rates.RANK_GATE_UP[rank] + fg_expert_rates.RANK_DOWN[rank]
                    for rank in range(8))
        self.assertEqual(fleet, 1_504_256_000)

    def test_reduce_and_swiglu_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_log(Path(directory))
            result = fg_expert_rates.build([str(path)])
        rows = {(row["rank"], row["scope"], row["kernel"]): row for row in result["kernels"]}
        reduce = rows[(0, "expert_reduce", "fg_moe_reduce.spv")]
        self.assertEqual(reduce["bytes"], 6 * (11 * 2560 * 4 + 40))
        swiglu = rows[(0, "expert_activation", "fg_swiglu.spv")]
        self.assertEqual(swiglu["bytes"], 6 * 3 * 10 * 640 * 4)


if __name__ == "__main__":
    unittest.main()
