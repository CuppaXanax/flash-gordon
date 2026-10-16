import importlib.util
import os
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("mtp_source", Path(__file__).with_name("mtp_source.py"))
mtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtp)


class SourceTests(unittest.TestCase):
    def test_q8_round_trip_and_zero(self):
        values = np.linspace(-3, 3, 64, dtype=np.float32)
        packed = np.frombuffer(mtp.q8_0(values), dtype=np.uint8).reshape(-1, 34)
        scales = packed[:, :2].copy().reshape(-1).view("<f2").astype(np.float32)
        decoded = (packed[:, 2:].view(np.int8) * scales[:, None]).reshape(-1)
        self.assertLess(float(np.max(np.abs(values - decoded))), .025)
        self.assertEqual(mtp.q8_0(np.zeros(32, dtype=np.float32)), bytes(34))

    def test_reject_ignored_range_without_reading_body(self):
        source = object.__new__(mtp.Source)
        source.base = "https://example.invalid/"
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {}
        with patch.object(mtp.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(ValueError):
                source.read_range("shard.safetensors", 0, 8)
        response.read.assert_not_called()

    def test_pack_preserves_raw_norms_and_q8_descriptor(self):
        path = Path(f".mtp-source-unit-{os.getpid()}.gguf")
        self.assertFalse(path.exists())
        norm = np.array([.5, -.5], dtype=np.float32)
        matrix = np.arange(32, dtype=np.float32) / 32
        buffers = [norm.view(np.uint32).__rshift__(16).astype("<u2").tobytes(),
                   matrix.view(np.uint32).__rshift__(16).astype("<u2").tobytes()]
        audit = dict(revision="a" * 40, tensors=[
            dict(name="mtp.norm.weight", shape=[2], bytes=4, shard="a", file_offset=0),
            dict(name="mtp.fc.weight", shape=[1, 32], bytes=64, shard="b", file_offset=0)])
        source = unittest.mock.Mock()
        source.read_range.side_effect = buffers
        try:
            mtp.pack(source, audit, path)
            data = path.read_bytes()
            self.assertEqual(data[:4], b"GGUF")
            self.assertEqual(struct.unpack_from("<IQQ", data, 4), (3, 2, 3))
            # Last tensor is at offset 32, 34 bytes; preceding vector is raw F32.
            data_start = len(data) - 32 - 34
            self.assertEqual(data_start % 32, 0)
            self.assertEqual(data[data_start:data_start + 8], norm.astype("<f4").tobytes())
            self.assertEqual(data[data_start + 32:], mtp.q8_0(matrix))
            self.assertTrue(audit["export"]["raw_norm_gammas"])
            with self.assertRaises(FileExistsError):
                mtp.pack(source, audit, path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
