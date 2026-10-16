import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import contextlib
import io
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import bf16_source as source

spec = importlib.util.spec_from_file_location(
    "bf16_analysis", Path(__file__).resolve().parents[1]/"tools"/"analyze_bf16_hops.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def receipt(self):
        import json
        rows = []
        for number in range(5):
            for ng in (False, True):
                for leg in range(3):
                    bf = leg != 1 if number % 2 else leg == 1
                    size = 24+(10240+(2560 if ng else 0))*(2 if bf else 4)
                    value = 600 if bf else 1000
                    rows.append(dict(round=number, leg=leg, ngram=ng, bf16=bf,
                        payload_bytes=size, frame_bytes=size+32, samples=500,
                        rtt_mean_us=value, rtt_p50_us=value, rtt_p99_us=value+10,
                        codec_validated=True))
        return [dict(IP="192.168.42.52", ExitCode=0, Stderr="",
                     Stdout="\n".join(map(json.dumps, rows))),
                dict(IP="192.168.42.53", ExitCode=0, Stderr="", Stdout="")]

    def test_paired_model(self):
        result = analysis.transport(self.receipt())
        self.assertEqual(result["ordinary"]["saving_us"]["mean"], 200)
        self.assertEqual(result["eight_work_hop_model"]["modeled_saving_us"]["mean"], 1600)
        self.assertEqual(result["eight_work_hop_model"]["modeled_saving_us"]["ci95_half"], 0)

    def test_incomplete_or_failed_pair(self):
        r = self.receipt();r[0]["Stdout"] = "\n".join(r[0]["Stdout"].splitlines()[:-1])
        with self.assertRaises(ValueError):analysis.transport(r)
        r = self.receipt();r[1]["ExitCode"] = 1
        with self.assertRaises(ValueError):analysis.transport(r)

    def test_quality_thresholds(self):
        rows = [dict(case=c, position=p, final=p == 3, vocab=248320, kl=.001,
                     teacher_top1=7, bf16_top1=7) for c in range(80) for p in range(4)]
        self.assertTrue(analysis.quality(rows)["primary_80"]["passed"])
        rows[3]["bf16_top1"] = 9
        self.assertTrue(analysis.quality(rows)["primary_80"]["passed"])
        rows[7]["bf16_top1"] = 9
        self.assertFalse(analysis.quality(rows)["primary_80"]["passed"])
        with self.assertRaises(ValueError):analysis.quality(rows[:-1])
        rows[-1]["position"] = 99
        with self.assertRaises(ValueError):analysis.quality(rows)

    def test_metadata_only_tokenizer_header(self):
        directory, overlay, public = MagicMock(), MagicMock(), MagicMock()
        public.revision = source.REVISION
        items = [dict(path=f"{i}.gguf", size=100) for i in range(2)]
        headers = [b"tokenizer metadata", b"tensor metadata"]
        def read(shard, offset, size):
            index = int(shard[0])
            return struct.pack("<4sIQQ", b"GGUF", 3, index, 0) if size == 24 else headers[index]
        public.read_range.side_effect = read
        saved = []
        header = MagicMock()
        header.write_bytes.side_effect = saved.append
        directory.__truediv__.side_effect = lambda name: header if name == "public-header.gguf" else MagicMock()
        with (patch.object(source, "Source", return_value=public),
              patch.object(source, "get_json", return_value=items),
              patch.object(source, "load_manifest", return_value=dict(sha256=source.PINNED_MANIFEST, records=[])),
              patch.object(source, "gguf_header", return_value=dict(records=[])),
              patch.object(source.hashlib, "file_digest") as digest,
              contextlib.redirect_stdout(io.StringIO())):
            digest.return_value.hexdigest.return_value = source.OVERLAY_SHA
            source.metadata(Path("unused"), directory, overlay)
        self.assertEqual(saved, [headers[0]])


if __name__ == "__main__":
    unittest.main()
