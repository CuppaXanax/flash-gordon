import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from q4_quantize import target
from q4_source import SIZES


class NativeQuantization(unittest.TestCase):
    def test_q5_1_block_size(self):
        self.assertEqual(SIZES[7], (32, 24))

    def test_supported_native_formats(self):
        cases = [
            ("fc_hidden.weight", [2560, 2560], 12),
            ("hyper_connection_mixer.input_mix_weight_up.weight", [320, 10240], 20),
            ("layers.0.mlp.experts.gate_up_proj", [2560, 1280, 512], 12),
            ("layers.0.mlp.experts.down_proj", [640, 2560, 512], 7),
            ("layers.0.mlp.gate.weight", [2560, 512], 8),
            ("layers.0.mlp.shared_expert_gate.weight", [2560, 1], 8),
            ("layers.0.attn_hyper_connection.block_inject_weight.weight", [10240, 4], 8),
            ("pre_fc_norm_hidden.weight", [10240], 0),
        ]
        for name, shape, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(target(dict(name="mtp." + name, shape=shape),
                                        False, mtp=True), expected)

    def test_rejects_target_weights(self):
        with self.assertRaises(ValueError):
            target(dict(name="blk.0.attn_q.weight", shape=[2560, 12288]), False, mtp=True)

    def test_preserves_common_projections(self):
        common = dict(name="mtp.fc_hidden.weight", shape=[2560, 2560])
        self.assertEqual(target(common, False, mtp=True, mtp_experts_only=True), 8)
        expert = dict(name="mtp.layers.0.mlp.experts.gate_up_proj", shape=[2560, 1280, 512])
        self.assertEqual(target(expert, False, mtp=True, mtp_experts_only=True), 12)

    def test_existing_target_selection_unchanged(self):
        self.assertEqual(target(dict(name="blk.0.attn_q.weight", shape=[2560, 12288]), False), 12)
        self.assertEqual(target(dict(name="blk.0.attn_q.weight", shape=[2560, 12288]), True), 13)


if __name__ == "__main__":
    unittest.main()
