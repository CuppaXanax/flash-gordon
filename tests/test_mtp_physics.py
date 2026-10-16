import importlib.util
from pathlib import Path
import unittest
import shutil
import struct
import uuid

spec = importlib.util.spec_from_file_location(
    "mtp_physics", Path(__file__).resolve().parents[1] / "tools" / "mtp_physics.py")
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)


class PhysicsTests(unittest.TestCase):
    def test_overlap_and_missing_positions(self):
        rows = [dict(category="chat", case="a", layer=6, token=t,
                     experts=list(range(t, t + 10))) for t in (0, 1, 2, 4)]
        result = physics.routing_unions(rows)
        self.assertEqual([r["windows"] for r in result], [4, 2, 1])
        self.assertEqual([r["mean_unique"] for r in result], [10, 11, 12])

    def test_streams_do_not_join(self):
        rows = [dict(category=c, case=c, layer=6, token=t,
                     experts=list(range(10))) for c, t in (("chat", 0), ("code", 1))]
        self.assertEqual(len(physics.routing_unions(rows)), 2)

    def test_duplicate_and_invalid(self):
        row = dict(category="chat", case="a", layer=6, token=0, experts=list(range(10)))
        with self.assertRaises(ValueError):
            physics.routing_unions([row, row])
        with self.assertRaises(ValueError):
            physics.routing_unions([dict(row, experts=[0] * 10)])

    def test_shared_bytes_and_bounds(self):
        weights = [dict(name="dense", copies=1, bytes_per_copy=100, layer=6),
                   dict(name="ffn_gate_exps.weight", copies=512,
                        bytes_per_copy=3, layer=6)]
        result = physics.summarize(weights, {n: [] for n in range(1, 5)})
        self.assertFalse(result["real_target_union_available"])
        self.assertEqual(result["dense_shared_bytes"], 100)
        self.assertEqual(result["shapes"][3]["bounds"][0]["bytes"], 130)
        self.assertEqual(result["shapes"][3]["bounds"][1]["bytes"], 220)

    def test_capture_anchors_and_missing_routes(self):
        root = Path(".build") / ("physics-test-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            (root / "cases.tsv").write_text("chat-00 chat 42 1 259 7\n")
            with self.assertRaisesRegex(ValueError, "no per-layer routing"):
                physics.captured_routes(root)
            (root / "routes-42-6.coverage").write_bytes(bytes([0]) + bytes([1]) * 259)
            ids = b"".join(struct.pack("<10I", *range(t % 2, t % 2 + 10))
                           for t in range(260))
            (root / "routes-42-6.u32").write_bytes(ids)
            rows = physics.captured_routes(root)
            result = physics.routing_unions(rows)
            self.assertEqual([r["windows"] for r in result], [256] * 4)
            self.assertEqual([r["mean_unique"] for r in result], [10, 11, 11, 11])
            (root / "routes-42-6.coverage").write_bytes(bytes(260))
            with self.assertRaisesRegex(ValueError, "hole"):
                physics.captured_routes(root)
        finally:
            shutil.rmtree(root)

    def test_cached_state_is_not_a_per_token_floor(self):
        weights = [dict(name="ssm_a", copies=1, bytes_per_copy=0, layer=6)]
        result = physics.summarize(weights, {n: [] for n in range(1, 5)})
        state = 48 * 128 * 128 * 4
        for shape in result["shapes"]:
            self.assertEqual(shape["cached_recurrent_minimum_bytes"], 2 * state)
            self.assertEqual(shape["bounds"][0]["bytes"], 2 * state)
            self.assertEqual(shape["unretained_shader_logical_state_bytes"],
                             3 * state * shape["n"])

    def test_non_target_provenance(self):
        weights = [dict(name="dense", copies=1, bytes_per_copy=100, layer=6)]
        rows = [dict(category="chat", case="a", layer=6, token=0, experts=list(range(10)))]
        result = physics.summarize(weights, {n: [] for n in range(1, 5)},
                                   routes=rows, routes_origin="native-draft")
        self.assertFalse(result["real_target_union_available"])


if __name__ == "__main__":
    unittest.main()
