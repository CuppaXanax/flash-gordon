import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import q4_source
import q4_quantize
import q4_quality


class Q4ToolsTest(unittest.TestCase):
    def test_header_and_native_name(self):
        md = q4_source.string("general.alignment")+struct.pack("<II", 4, 32)
        info = (q4_source.string("blk.0.attn_qkv.weight")+struct.pack("<IQQIQ", 2,
                2560, 10240, 30, 64))
        data = b"GGUF"+struct.pack("<IQQ", 3, 1, 1)+md+info
        with patch.object(q4_source, "fetch", return_value=data):
            metadata, tensors, base, alignment = q4_source.header("https://test")
        self.assertEqual(metadata, [md])
        self.assertEqual(tensors[0]["shape"], [2560, 10240])
        self.assertEqual(tensors[0]["offset"], 64)
        self.assertEqual(alignment, 32)
        self.assertEqual(base % 32, 0)

    def test_selected_exclusions(self):
        for name in ("blk.0.ffn_gate_exps.weight", "per_layer_token_embd.weight",
                     "mtp.weight", "v.patch_embd.weight"):
            self.assertFalse(q4_source.selected(name))
        self.assertTrue(q4_source.selected("blk.1.ple_value.weight"))

    def test_quantization_plan(self):
        cases = [
            ("blk.0.attn_qkv.weight", [2560, 10240], 12, 13),
            ("blk.0.hc_attn_up.weight", [320, 10240], 20, 20),
            ("blk.0.ffn_down_shexp.weight", [640, 2560], 20, 20),
            ("blk.0.ffn_gate_inp.weight", [2560, 512], 0, 0),
            ("blk.0.hc_attn_inject.weight", [10240, 4], 0, 0),
            ("blk.0.ssm_alpha.weight", [2560, 48], 0, 0),
            ("blk.3.indexer.q_proj.weight", [2560, 512], 30, 30),
            ("output.weight", [2560, 248320], 14, 14),
            ("token_embd.weight", [2560, 248320], 8, 8),
        ]
        for name, shape, normal, protected in cases:
            t = dict(name=name, shape=shape)
            self.assertEqual(q4_quantize.target(t, False), normal)
            self.assertEqual(q4_quantize.target(t, True), protected)

    def test_bf16_to_f32(self):
        result = q4_quantize.floats(struct.pack("<HHH", 0x3f80, 0xbf80, 0x0000), 30)
        self.assertEqual(struct.unpack("<fff", result), (1., -1., 0.))

    def test_lossless_router_plan(self):
        t = dict(name="blk.0.ffn_gate_inp.weight", shape=[2560, 512])
        self.assertEqual(q4_quantize.target(t, False, True), 30)
        self.assertEqual(q4_quantize.target(t, False, False), 0)

    def test_full_vocabulary_framing(self):
        raw = b"FGLOG01\n"+struct.pack("<I", 248320)+bytes(248320*4)
        with patch.object(Path, "read_bytes", return_value=raw):
            self.assertEqual(len(q4_quality.read(Path("unused.fgl"))), 248320)
        for invalid in (raw[:-4], raw[:12]+struct.pack("<f", float("nan"))+raw[16:],
                        b"FGLOG01\n"+struct.pack("<I", 1)+bytes(4)):
            with patch.object(Path, "read_bytes", return_value=invalid):
                with self.assertRaises(ValueError):
                    q4_quality.read(Path("unused.fgl"))

    def test_kld_and_shift(self):
        t = [1000., 999., -1000.]
        self.assertEqual(q4_quality.compare(t, t), (0., True))
        loss, top = q4_quality.compare(t, [x+10000 for x in t])
        self.assertLess(loss, 1e-10)
        self.assertTrue(top)
        loss, top = q4_quality.compare([4., 0.], [0., 4.])
        self.assertGreater(loss, 3.)
        self.assertFalse(top)
        self.assertEqual(len(list(q4_quality.contexts())), 80)

    @unittest.skipUnless(os.environ.get("FG_Q4_GGML_LIBRARY"), "requires built ggml library")
    def test_native_quantizer(self):
        scratch = Path(".q4-quantizer-test")
        scratch.mkdir()
        try:
            specs = [
                ("blk.0.attn_qkv.weight", 256, 130, 30, None),
                ("blk.0.hc_attn_up.weight", 320, 130, 30, None),
                ("output.weight", 256, 130, 30, None),
                ("blk.0.ffn_gate_inp.weight", 256, 2, 0, 0x3fc00000),
                ("blk.1.ffn_gate_inp.weight", 256, 2, 0, 0x3f800001),
            ]
            payload = bytearray()
            entries = []
            for name, width, rows, kind, value in specs:
                payload.extend(bytes((-len(payload)) % 32))
                entries.append(q4_source.string(name)+struct.pack("<IQQIQ", 2,
                               width, rows, kind, len(payload)))
                if kind == 30:
                    payload.extend(struct.pack("<H", 0x3f80)*(width*rows))
                else:
                    payload.extend(struct.pack("<I", value)*(width*rows))
            header = b"GGUF"+struct.pack("<IQQ", 3, len(entries), 0)+b"".join(entries)
            header += bytes((-len(header)) % 32)
            source = scratch/"source.gguf"
            source.write_bytes(header+payload)
            name = "blk.0.attn_qkv.weight"
            im_entries = [
                q4_source.string(name+".in_sum2")+struct.pack("<IQIQ", 1, 256, 0, 0),
                q4_source.string(name+".counts")+struct.pack("<IQIQ", 1, 1, 0, 1024),
            ]
            im_header = b"GGUF"+struct.pack("<IQQ", 3, 2, 0)+b"".join(im_entries)
            im_header += bytes((-len(im_header)) % 32)
            imatrix = scratch/"imatrix.gguf"
            imatrix.write_bytes(im_header+struct.pack("<256f", *(
                2.+i%7 for i in range(256)))+struct.pack("<f", 2.))
            for protected in (False, True):
                output = scratch/("protected.gguf" if protected else "ordinary.gguf")
                command = [sys.executable, str(ROOT/"q4_quantize.py"),
                           "--source", str(source), "--output", str(output),
                           "--ggml-library", os.environ["FG_Q4_GGML_LIBRARY"],
                           "--imatrix", str(imatrix)]
                if protected:
                    command.append("--protected")
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                serial = scratch/"serial.gguf"
                subprocess.run(command+["--threads", "1", "--output", str(serial)],
                               check=True, stdout=subprocess.DEVNULL)
                self.assertEqual(output.read_bytes(), serial.read_bytes())
                metadata, tensors, base, _ = q4_source.header(str(output))
                self.assertEqual([t["type"] for t in tensors],
                                 [13 if protected else 12, 20, 14, 30, 0])
                expected = q4_source.string("general.file_type")+struct.pack(
                    "<II", 4, 17 if protected else 15)
                self.assertIn(expected, metadata)
                ledger = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(ledger["imatrix"]["mapped"], 1)
                self.assertEqual(len(ledger["imatrix"]["missing"]), 2)
                data = output.read_bytes()
                for t in ledger["tensors"]:
                    raw = data[base+t["offset"]:base+t["offset"]+t["bytes"]]
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), t["sha256"])
                self.assertTrue(ledger["tensors"][3]["lossless_bf16"])
                self.assertFalse(ledger["tensors"][4]["lossless_bf16"])
                last = ledger["tensors"][4]
                self.assertEqual(data[base+last["offset"]:],
                                 struct.pack("<I", 0x3f800001)*512)
        finally:
            shutil.rmtree(scratch)


if __name__ == "__main__":
    unittest.main()
