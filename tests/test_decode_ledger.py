import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "decode_ledger", Path(__file__).resolve().parents[1] / "tools" / "decode-ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


class DecodeLedgerTests(unittest.TestCase):
    def capture(self, rank, tokens=(60, 61)):
        blocks = [
            (f"RING_DECODE token={token} embed_ms=1 ngram_ms=2 send_ms=3 "
             "own_ms=4 total_ms=44" if rank == 0 else
             f"RING_DECODE_BLOCK rank={rank} token={token} total_ms=3")
            for token in tokens
        ]
        for wait in (1000, 2):
            blocks.append(json.dumps({
                "schema": "flash-gordon.profile", "record_type": "fabric_service",
                "message_type": 44, "direction": "receive", "wait_ms": wait,
            }, separators=(",", ":")))
        blocks.extend([
            "DECODE_PROFILE rank=0 gpu_ms=4",
            "DECODE_PROFILE_KERNEL scope=expert gpu_ms=1",
            "DECODE_PROFILE_KERNEL scope=expert gpu_ms=2",
            "DECODE_PROFILE rank=0 gpu_ms=5",
            "DECODE_PROFILE_KERNEL scope=expert gpu_ms=5",
        ])
        return "\n".join(blocks)

    def summarize(self, captures, **kwargs):
        def read(path):
            rank = int(path.name.split("-")[1])
            return captures[rank]
        with patch.object(Path, "read_text", read):
            return ledger.summarize(Path("."), "test", 2, **kwargs)

    def test_additive_ledger_and_per_block_kernel_sums(self):
        result = self.summarize([self.capture(rank) for rank in range(8)])
        self.assertEqual(result["serial_ledger_ms"], {
            "coordinator_prepare": 6, "rank_services_including_enqueue_hc": 25,
            "transport_head_completion_and_unattributed": 13, "observed": 44,
        })
        self.assertEqual(result["ranks"][0]["kernel_scopes_ms"]["expert"], 4)
        self.assertEqual(result["token_start"], 60)
        self.assertEqual(result["token_end"], 61)

    def test_legacy_embed_does_not_double_count_preparation(self):
        result = self.summarize([self.capture(rank) for rank in range(8)], legacy_embed=True)
        self.assertEqual(result["serial_ledger_ms"]["coordinator_prepare"], 1)
        self.assertEqual(result["serial_ledger_ms"]["transport_head_completion_and_unattributed"], 18)

    def test_rejects_misaligned_windows(self):
        captures = [self.capture(rank) for rank in range(8)]
        captures[3] = self.capture(3, (61, 62))
        with self.assertRaisesRegex(ValueError, "differs from coordinator"):
            self.summarize(captures)

    def test_rejects_incomplete_windows(self):
        captures = [self.capture(rank) for rank in range(8)]
        captures[7] = self.capture(7, (60,))
        with self.assertRaisesRegex(ValueError, "consecutive decode tokens"):
            self.summarize(captures)


if __name__ == "__main__":
    unittest.main()
