#!/usr/bin/env python3
"""Unit tests for tools/fg_dense_rates.py (offline, no Vulkan)."""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fg_dense_rates  # noqa: E402

SAMPLE = "\n".join([
    "DECODE_PROFILE_KERNEL rank=0 scope=gdn_output kernel=fg_dense_q8_0_cooked_r8.spv calls=4 gpu_ms=0.382",
    "DECODE_PROFILE_KERNEL rank=0 scope=gr_attn_read kernel=fg_group_rms_norm.spv calls=6 gpu_ms=0.087",
    "DECODE_PROFILE_KERNEL rank=0 scope=gr_attn_read kernel=fg_dense_q8_0_cooked_split_reduce_silu.spv calls=6 gpu_ms=0.021",
    "DECODE_PROFILE_KERNEL rank=1 scope=qsa_projection kernel=fg_dense_q8_0_cooked_r8.spv calls=6 gpu_ms=0.262",
    "unrelated log line",
]) + "\n"


def write_log(directory: Path) -> Path:
    path = directory / "profile.log"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


class DenseRatesTest(unittest.TestCase):
    def test_rows_and_rates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = write_log(Path(directory))
            capture = io.StringIO()
            with redirect_stdout(capture):
                status = fg_dense_rates.main([str(path), "--json"])
            self.assertEqual(status, 0)
            result = json.loads(capture.getvalue())
        rows = {(row["rank"], row["scope"], row["kernel"]): row for row in result["kernels"]}
        output = rows[(0, "gdn_output", "fg_dense_q8_0_cooked_r8.spv")]
        self.assertEqual(output["calls"], 4)
        self.assertGreater(output["achieved_gbps"], 170.0)
        self.assertLess(output["achieved_gbps"], 180.0)
        rms = rows[(0, "gr_attn_read", "fg_group_rms_norm.spv")]
        self.assertEqual(rms["bytes"], 6 * 3 * 10240 * 4)
        unpriced = rows[(1, "qsa_projection", "fg_dense_q8_0_cooked_r8.spv")]
        self.assertFalse(unpriced["priced"])
        self.assertIsNone(unpriced["achieved_gbps"])

    def test_cooked_tile_size(self):
        self.assertEqual(fg_dense_rates.cooked_matrix(2560, 10240), 640 * (2560 + 40960))
        self.assertEqual(fg_dense_rates.cooked_matrix(6144, 2560), 160 * (6144 + 98304))


if __name__ == "__main__":
    unittest.main()
