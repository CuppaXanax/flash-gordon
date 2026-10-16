import hashlib
import pathlib
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from mtp_format_audit import gguf_header, load_manifest, target_support


class FormatAuditTest(unittest.TestCase):
    def test_raw_manifest_abi(self):
        start = 184 + 8 * 120 + 48 + 48 * 4 + 48 * 512 * 2
        raw = bytearray(start + 200)
        raw[:8] = b"QOWNFGM1"
        struct.pack_into("<6I", raw, 8, 6, 6, len(raw), 8, 48, 512)
        struct.pack_into("<I", raw, 60, 1)
        name = b"blk.2.ffn_down_exps.weight.rank1"
        raw[start:start + len(name)] = name
        struct.pack_into("<QQII4QHHHBB", raw, start + 96,
                         4096, 891289600, 8, 3, 640, 2560, 512, 0, 1, 2, 65535, 2, 5)
        raw[start + 164:start + 196] = b"x" * 32
        with mock.patch.object(pathlib.Path, "read_bytes", return_value=bytes(raw)):
            result = load_manifest("unused")
        self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["records"][0]["shape"], [640, 2560, 512])
        self.assertEqual(result["records"][0]["sha256"], (b"x" * 32).hex())
        self.assertEqual(target_support(result["records"][0])[0], "ffn_down_exps.weight")
        raw[8] = 5
        with mock.patch.object(pathlib.Path, "read_bytes", return_value=bytes(raw)):
            with self.assertRaisesRegex(ValueError, "ABI"):
                load_manifest("unused")

    def test_unknown_format_rejected(self):
        record = dict(name="blk.2.ffn_down_exps.weight.rank1", type=8, layout=5,
                      kind=2, dims=3, shape=[640, 2560, 512])
        target_support(record)
        for changes in (dict(type=6), dict(layout=0), dict(shape=[640, 2560, 128]),
                        dict(name="new_unknown.weight")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                target_support(dict(record, **changes))

    def test_bf16_and_dense_families(self):
        for name, kind, layout in (("ffn_gate_inp.weight", 30, 0),
                                   ("indexer.k_proj.weight", 30, 0),
                                   ("ffn_down_shexp.weight", 20, 7),
                                   ("ffn_gate_shexp.weight", 12, 6),
                                   ("ssm_out.weight", 13, 7),
                                   ("attn_k.weight", 8, 0)):
            family, route = target_support(dict(name="blk.3." + name, type=kind,
                                                layout=layout, kind=1))
            self.assertEqual(family, name)
            self.assertTrue(route)

    def test_gguf_arrays_offsets_and_truncation(self):
        def string(text):
            data = text.encode()
            return struct.pack("<Q", len(data)) + data
        raw = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
        raw += string("general.alignment") + struct.pack("<II", 4, 64)
        raw += string("ignored.array") + struct.pack("<IIQ", 9, 8, 2)
        raw += string("first") + string("second")
        raw += string("weight") + struct.pack("<IQQIQ", 2, 32, 16, 8, 0)
        raw += bytes(-len(raw) % 64)
        result = gguf_header(raw)
        self.assertEqual(result["records"][0]["bytes"], 544)
        self.assertEqual(result["records"][0]["file_offset"], len(raw))
        self.assertEqual(result["data_offset"] % 64, 0)
        with self.assertRaises(ValueError):
            gguf_header(raw[:40])


if __name__ == "__main__":
    unittest.main()
