import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]/"tools"))
SPEC = importlib.util.spec_from_file_location("mtp_live_analyze",
    Path(__file__).parents[1]/"tools"/"mtp_live_analyze.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiveAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(".build")/f"mtp-live-analysis-{os.getpid()}"
        self.root.mkdir()
        (self.root/"outcome.json").write_text(json.dumps(dict(
            productionVerified=True, scratchRemoved=True, complete=True, diagnosticOnly=True)))
        for phase, tps in (("A0",30),("F3",40),("A1",30)):
            rows = [dict(id=kind,case=kind,repeat=0,category=kind,decode_tps=tps,
                         wall_tps=tps/2) for kind in ("chat","code")]
            (self.root/f"{phase}-perf.json").write_text(json.dumps(dict(rows=rows)))
        trace = ""
        for kind in ("chat","code"):
            trace += ("MTP_ROUND slot=0 first=1 k=3 accepted=3 emitted=4 consumed=4 ms=35\n"
                "MTP_STAGE slot=0 first=1 k=3 draft_ms=10 prepare_ms=2 verify_ms=20 "
                "target_repair_ms=1 native_repair_ms=0.5 commit_ms=1 host_ms=2.5 "
                "total_ms=35 overlap_ms=2 target_replay=0\n"
                "MTP_CHOICE emitted=3 target_argmax=3\n"
                f"API_TOKEN_HASH id={kind} count=4 sha256={'a'*64}\n")
        (self.root/"F3-rank-0.log").write_text(trace)
        for rank in range(8):
            log = (f"MTP_VERIFY_RANK rank={rank} first=1 n=4 replay=0 "
                "input_ms=0.1 compute_ms=1.8 read_ms=0.1 forward_or_head_ms=0 "
                "total_ms=2 status=0\n")*2
            raw = log.encode()
            with tarfile.open(self.root/f"evidence-192.168.42.{42+rank}.tar.gz","w:gz") as archive:
                member = tarfile.TarInfo(f"serving-F3-rank-{rank}.log")
                member.size = len(raw)
                archive.addfile(member,io.BytesIO(raw))
        rows = [dict(round=r,n=n,initial=initial,rtt_mean_us=1000*n,
                     frame_bytes=41000*n,samples=100,validated=True)
                for r in range(5) for n in range(1,5) for initial in (False,True)]
        (self.root/"transport.json").write_text(json.dumps([dict(Stdout="\n".join(map(json.dumps,rows)))]))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_repeated_prefixes_keep_request_alignment(self):
        report = MODULE.analyze(self.root,self.root/"transport.json")
        self.assertAlmostEqual(report["comparisons"]["code"]["n4_owner_compute_sum_ms"]["mean"],14.4)
        self.assertAlmostEqual(report["comparisons"]["chat"]["n4_transport_head_host_residual_ms"]["mean"],4)
        self.assertFalse(report["gates"]["promotion"])
        self.assertEqual(report["direct_codec_crc_tcp_pair"]["4"]["eight_work_hop_pair_transfer_model_ms"],16)

    def test_unrestored_outcome_is_not_a_report(self):
        (self.root/"outcome.json").write_text(json.dumps(dict(productionVerified=False)))
        with self.assertRaises(AssertionError):
            MODULE.analyze(self.root,self.root/"transport.json")

    def test_incomplete_mode_does_not_invent_performance(self):
        (self.root/"outcome.json").write_text(json.dumps(dict(
            productionVerified=True,scratchRemoved=True,complete=False,diagnosticOnly=True)))
        for rank in range(8):
            raw = (f"MTP_VERIFY_RANK rank={rank} first=1 n=4 replay=0 "
                "input_ms=0.1 compute_ms=1.8 read_ms=0.1 status=0\n").encode()
            with tarfile.open(self.root/f"evidence-192.168.42.{42+rank}.tar.gz","w:gz") as archive:
                member = tarfile.TarInfo(f"serving-F3-rank-{rank}.log")
                member.size = len(raw)
                archive.addfile(member,io.BytesIO(raw))
        report = MODULE.analyze_incomplete(self.root,self.root/"transport.json")
        self.assertIsNone(report["mtp_decode_tps"])
        self.assertIsNone(report["live_acceptance"])
        self.assertAlmostEqual(report["cold_verify_compute_sum_ms"],14.4)
