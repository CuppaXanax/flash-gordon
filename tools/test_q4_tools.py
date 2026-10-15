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
import q4_compare_source
import q4_mix


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

    def test_source_comparison(self):
        scratch = Path(".q4-source-test")
        scratch.mkdir()
        try:
            info = q4_source.string("blk.0.ssm_out.weight")+struct.pack("<IQQIQ",2,32,1,8,0)
            h = b"GGUF"+struct.pack("<IQQ",3,1,0)+info
            h += bytes((-len(h))%32)
            a, b = scratch/"a.gguf", scratch/"b.gguf"
            a.write_bytes(h+struct.pack("<e",.5)+bytes(range(32)))
            b.write_bytes(a.read_bytes())
            self.assertEqual(q4_compare_source.compare(a,b)["identical"],1)
            b.write_bytes(h+struct.pack("<e",1.)+bytes(range(32)))
            result = q4_compare_source.compare(a,b)
            self.assertEqual(result["different"],1)
            self.assertEqual(result["rows"][0]["scale_ratio_median"],.5)
            self.assertEqual(result["rows"][0]["identical_code_blocks"],1)
            b.write_bytes(b.read_bytes()[:-1])
            with self.assertRaises(ValueError): q4_compare_source.compare(a,b)
        finally:
            shutil.rmtree(scratch)

    def test_exact_mixed_plan(self):
        scratch=Path(".q4-mix-test");scratch.mkdir()
        specs=[("blk.0.attn_qkv.weight",256),("blk.0.hc_attn_up.weight",320)]
        paths={}
        try:
            for policy in ("q4","q5","q6","q8"):
                entries,payload=[],bytearray()
                for index,(name,width) in enumerate(specs):
                    kind=8 if policy=="q8" else 20 if width%256 else {"q4":12,"q5":13,"q6":14}[policy]
                    payload.extend(bytes((-len(payload))%32))
                    entries.append(q4_source.string(name)+struct.pack("<IQQIQ",2,width,1,kind,len(payload)))
                    block,size=q4_source.SIZES[kind]
                    payload.extend(bytes([index+kind])*(width//block*size))
                h=b"GGUF"+struct.pack("<IQQ",3,2,0)+b"".join(entries)
                paths[policy]=scratch/f"{policy}.gguf"
                paths[policy].write_bytes(h+bytes((-len(h))%32)+payload)
            output=scratch/"identical.gguf"
            q4_mix.mix(paths,{},output)
            self.assertEqual(output.read_bytes(),paths["q8"].read_bytes())
            output=scratch/"mixed.gguf"
            report=q4_mix.mix(paths,{"families":{"gdn-qkv":"q5","hc":"q5"}},output)
            self.assertEqual([t["type"] for t in q4_source.header(str(output))[1]],[13,8])
            self.assertEqual(report["changed_families"],["gdn-qkv"])
            self.assertEqual(report["tensors"][1]["policy"],"q8")
            output=scratch/"changed-families.gguf"
            report=q4_mix.mix(paths,{"families":{"gdn-qkv":"q5","hc":"q5"}},output,True)
            self.assertEqual([t["name"] for t in q4_source.header(str(output))[1]],[specs[0][0]])
            self.assertEqual(report["changed_families"],["gdn-qkv"])
            with self.assertRaises(ValueError):
                q4_mix.mix(paths,{},scratch/"empty.gguf",True)
            routers=scratch/"routers.gguf"
            name="blk.0.ffn_gate_inp.weight"
            h=b"GGUF"+struct.pack("<IQQ",3,1,0)+q4_source.string(name)+struct.pack(
                "<IQQIQ",2,2,1,30,0)
            payload=struct.pack("<HH",0x3fc0,0x3e80)
            routers.write_bytes(h+bytes((-len(h))%32)+payload)
            output=scratch/"with-routers.gguf"
            report=q4_mix.mix(dict(paths,routers=routers),
                {"families":{"gdn-qkv":"q4"}},output,True)
            self.assertEqual(report["bf16_router_count"],1)
            self.assertEqual([t["type"] for t in q4_source.header(str(output))[1]],[12,30])
            self.assertEqual(report["changed_families"],["gdn-qkv"])
            self.assertTrue(output.read_bytes().endswith(payload))
            output=scratch/"mixed-q6.gguf"
            report=q4_mix.mix(paths,{"families":{"gdn-qkv":"q6","hc":"q6"}},output)
            self.assertEqual([t["type"] for t in q4_source.header(str(output))[1]],[14,8])
            self.assertEqual(report["tensors"][1]["policy"],"q8")
            with self.assertRaises(ValueError):
                q4_mix.mix(paths,{"families":{"hc":"q3"}},scratch/"invalid.gguf")
            with self.assertRaises(ValueError):
                q4_mix.mix(paths,{"tensors":{"unknown":"q4"}},scratch/"invalid.gguf")
        finally:
            shutil.rmtree(scratch)

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
            q5 = scratch/"q5.gguf"
            subprocess.run([sys.executable,str(ROOT/"q4_quantize.py"),
                "--source",str(source),"--output",str(q5),
                "--ggml-library",os.environ["FG_Q4_GGML_LIBRARY"],
                "--q5-dense"],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual([t["type"] for t in q4_source.header(str(q5))[1]],[13,20,14,30,0])
            q6 = scratch/"q6.gguf"
            subprocess.run([sys.executable,str(ROOT/"q4_quantize.py"),
                "--source",str(source),"--output",str(q6),
                "--ggml-library",os.environ["FG_Q4_GGML_LIBRARY"],
                "--q6-dense"],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual([t["type"] for t in q4_source.header(str(q6))[1]],[14,20,14,30,0])
            self.assertIn(q4_source.string("general.file_type")+struct.pack("<II",4,18),
                          q4_source.header(str(q6))[0])
            control = scratch/"control.gguf"
            subprocess.run([sys.executable,str(ROOT/"q4_quantize.py"),
                "--source",str(source),"--output",str(control),
                "--ggml-library",os.environ["FG_Q4_GGML_LIBRARY"],
                "--q8-control"],check=True,stdout=subprocess.DEVNULL)
            metadata, tensors, _, _ = q4_source.header(str(control))
            self.assertEqual([t["type"] for t in tensors],[8,8,8,30,0])
            self.assertIn(q4_source.string("general.file_type")+struct.pack("<II",4,7),metadata)
            roundtrip = scratch/"q8-roundtrip.gguf"
            subprocess.run([sys.executable,str(ROOT/"q4_quantize.py"),
                "--source",str(control),"--output",str(roundtrip),
                "--ggml-library",os.environ["FG_Q4_GGML_LIBRARY"],
                "--q8-control"],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(control.read_bytes(),roundtrip.read_bytes())
            _, refs, base, _ = q4_source.header(str(control))
            ref = refs[0]
            size = ref["shape"][0]*ref["shape"][1]//32*34
            payload = control.read_bytes()[base+ref["offset"]:base+ref["offset"]+size]
            mapping = scratch/"overrides.json"
            row = dict(name=ref["name"],shape=ref["shape"],identical=False,
                       production_sha256=hashlib.sha256(payload).hexdigest())
            mapping.write_text(json.dumps(dict(rows=[row])))
            aligned = scratch/"aligned.gguf"
            command = [sys.executable,str(ROOT/"q4_quantize.py"),
                "--source",str(source),"--output",str(aligned),
                "--ggml-library",os.environ["FG_Q4_GGML_LIBRARY"],"--q8-control",
                "--q8-overrides",str(control),"--override-map",str(mapping)]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(control.read_bytes(),aligned.read_bytes())
            self.assertEqual(json.loads(aligned.with_suffix(".json").read_text())[
                "q8_source_override_count"],1)
            dense_reference = scratch/"dense-reference.gguf"
            h = b"GGUF"+struct.pack("<IQQ",3,1,0)+q4_source.string(ref["name"])+struct.pack(
                "<IQQIQ",2,*ref["shape"],8,0)
            dense_reference.write_bytes(h+bytes((-len(h))%32)+payload)
            dense = scratch/"dense-only.gguf"
            subprocess.run(command+["--q8-overrides",str(dense_reference),
                "--deployed-dense-only","--output",str(dense)],check=True,
                stdout=subprocess.DEVNULL)
            self.assertEqual([t["name"] for t in q4_source.header(str(dense))[1]],[ref["name"]])
            row["production_sha256"] = "0"*64
            mapping.write_text(json.dumps(dict(rows=[row])))
            self.assertNotEqual(subprocess.run(command,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL).returncode,0)
        finally:
            shutil.rmtree(scratch)


if __name__ == "__main__":
    unittest.main()
