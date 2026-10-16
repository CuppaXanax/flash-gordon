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
import analyze_fp16_hops as fp16
import fp16_q8_restore as q8_restore

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

    def test_scaled_transport_overhead(self):
        import json
        for codec in ("fp16","fp16-128"):
            receipt=self.receipt();rows=[]
            for line in receipt[0]["Stdout"].splitlines():
                row=json.loads(line);compressed=row["bf16"]
                count=10240+(2560 if row["ngram"] else 0)
                overhead=(4*(count//128) if codec=="fp16-128" else 4) if compressed else 0
                row.update(compressed=compressed,codec=codec if compressed else "f32",bf16=False)
                row["payload_bytes"]+=overhead;row["frame_bytes"]+=overhead
                rows.append(row)
            receipt[0]["Stdout"]="\n".join(map(json.dumps,rows))
            result=fp16.scaled_transport(receipt,codec)
            self.assertEqual(result["eight_work_hop_model"]["modeled_saving_us"]["mean"],1600)
            self.assertEqual(result["work_frames_fp16_bytes"],169408+(2640 if codec=="fp16-128" else 32))
            if codec=="fp16":
                for policy in ("mixed_first_f32_model","mixed_no_rank0_model"):
                    self.assertEqual(result[policy]["saving_us"]["mean"],1400)
                    self.assertEqual(result[policy]["saving_us"]["ci95_half"],0)
            rows[0]["codec"]="wrong";receipt[0]["Stdout"]="\n".join(map(json.dumps,rows))
            with self.assertRaises(AssertionError):fp16.scaled_transport(receipt,codec)

    def test_cumulative_kl_not_additive(self):
        teacher=[0.,1.,-1.];mixed=[.1,1.1,-1.2];candidate=[.01,1.01,-1.]
        base,_=fp16.compare(teacher,mixed)
        delta,_=fp16.compare(mixed,candidate)
        cumulative,_=fp16.compare(teacher,candidate)
        self.assertGreater(delta,0)
        self.assertLess(cumulative,base)
        self.assertNotAlmostEqual(cumulative,base+delta)

    def test_q8_restore_preserves_existing_scratch(self):
        args=MagicMock();args.rank=0
        args.output.exists.return_value=False
        scratch=args.output.with_suffix.return_value
        scratch.exists.return_value=True
        row=dict(name="tensor",rank=0)
        with (patch.object(q8_restore,"load_manifest",side_effect=[
                dict(sha256=q8_restore.ORIGINAL,records=[row]),
                dict(sha256=q8_restore.MIXED,records=[row])]),
              patch.object(q8_restore,"digest",return_value=q8_restore.DENSE),
              patch.object(q8_restore,"gguf_header",return_value=dict(records=[]))):
            with self.assertRaisesRegex(ValueError,"scratch already exists"):
                q8_restore.restore(args)
        args.output.open.assert_not_called()
        scratch.write_bytes.assert_not_called()
        scratch.unlink.assert_not_called()

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
