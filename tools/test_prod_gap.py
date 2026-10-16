import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch


def module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


probe = module("prod-gap-probe")
analyze = module("analyze-prod-gap")


class ProbeTests(unittest.TestCase):
    def test_actual_one_token_not_requested_ceiling(self):
        payload = {"usage": {"completion_tokens": 1, "prompt_tokens": 5},
                   "choices": [{"message": {"content": "OK"}}]}
        response = io.BytesIO(json.dumps(payload).encode())
        response.headers = {"X-Flash-Gordon-Decode-TPS": "10"}
        with patch.object(probe.urllib.request, "urlopen", return_value=response) as opened:
            result = probe.request([{"role": "user", "content": "OK"}], 128)
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual(result["usage"]["completion_tokens"], 1)
        self.assertGreater(result["client_wall_tps"], 0)

    def test_interval_union_does_not_double_count(self):
        self.assertEqual(analyze.union_ms([(0, 2_000_000), (1_000_000, 3_000_000)]), 3)

    def test_cpu_and_engine_accounting_are_separate(self):
        calls = [
            {"op": "submit", "begin_ns": "0", "end_ns": "1000000",
             "object": "a", "result": "0", "gpu_before_ns": "2000000"},
            {"op": "wait", "begin_ns": "1000000", "end_ns": "5000000",
             "object": "b", "result": "0", "gpu_after_ns": "5000000"},
        ]
        result = analyze.cpu_calls(calls)
        self.assertEqual(result["submit_ms"], 1)
        self.assertEqual(result["wait_ms"], 4)
        self.assertEqual(result["gpu_engine_ms"], 3)
        self.assertEqual(result["accounted_union_ms"], 5)

    def test_native_embedding_label_is_not_real_embedding(self):
        rows = analyze.parse_lines([
            "RING_DECODE token=12 embed_ms=3 ngram_ms=2 send_ms=.5 own_layers=6..11 own_ms=4 total_ms=20\n",
            "GENERATE_TRACE tokens=1 decode_ms=20 total_ms=20\n",
        ], 0)
        row = rows["requests"][0]["rings"][0]["native"]
        self.assertEqual(float(row["embed_ms"])-float(row["ngram_ms"])-float(row["send_ms"]), .5)

    def test_missing_calls_are_not_zero_gpu_or_host_time(self):
        self.assertEqual(analyze.cpu_calls([]), {"counts": {}})

    def test_unavailable_drm_counter_is_not_gpu_time(self):
        unknown = str(2**64-1)
        calls = [
            {"op": "submit", "begin_ns": "0", "end_ns": "1",
             "object": "a", "result": "0", "gpu_before_ns": unknown},
            {"op": "wait", "begin_ns": "1", "end_ns": "2",
             "object": "b", "result": "0", "gpu_after_ns": unknown},
        ]
        self.assertNotIn("gpu_engine_ms", analyze.cpu_calls(calls))

    def test_coordinator_has_two_work_sends_not_one(self):
        def event(direction, wall):
            return json.dumps({"schema": "flash-gordon.profile",
                "record_type": "fabric_service", "message_type": 44,
                "direction": direction, "wall_ms": wall}, separators=(",", ":")) + "\n"
        rows = analyze.parse_lines([
            event("send", .1), event("receive", .2),
            "VKWALL op=submit begin_ns=1 end_ns=2 object=a result=0\n",
            event("send", .03),
            "RING_DECODE token=12 own_ms=4 total_ms=20\n",
            "GENERATE_TRACE tokens=1 decode_ms=20 total_ms=20\n",
        ], 0)
        row = rows["rings"][0]
        self.assertEqual(row["initial_send"]["wall_ms"], .1)
        self.assertEqual(row["send"]["wall_ms"], .03)
        self.assertEqual(row["receive"]["wall_ms"], .2)
        self.assertEqual(row["cpu"]["counts"]["submit"], 1)

    def test_replay_weights_36_gdn_12_qsa_not_eight_identical_blocks(self):
        row = analyze.replay_round(900, [100, 200, 100, 100, 100, 200])
        self.assertEqual(row["type_weighted_layer_ms"], 6)
        self.assertEqual(row["unassigned_replay_remainder_per_block_us"], 100)
        self.assertEqual(row["type_weighted_wall_reference_ms"], 6.8)

    def test_head_calls_are_not_charged_to_block(self):
        def event(direction):
            return json.dumps({"schema": "flash-gordon.profile",
                "record_type": "fabric_service", "message_type": 44,
                "direction": direction}, separators=(",", ":")) + "\n"
        rows = analyze.parse_lines([
            event("receive"),
            "VKWALL op=submit begin_ns=1 end_ns=2 object=a result=0\n",
            event("send"),
            "RING_DECODE_BLOCK token=12 total_ms=1\n",
            "VKWALL op=submit begin_ns=3 end_ns=4 object=a result=0\n",
            "VKWALL op=wait begin_ns=4 end_ns=5 object=b result=0\n",
            "OUTPUT_SPLIT_STATE rank=1 token=12 what=slice detail=1\n",
        ], 1)
        self.assertEqual(rows["blocks"][0]["cpu"]["counts"], {"submit": 1})
        self.assertEqual(rows["heads"][0]["cpu"]["counts"], {"submit": 1, "wait": 1})


if __name__ == "__main__":
    unittest.main()
