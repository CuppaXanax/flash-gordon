import copy
import unittest

from analyze_mtp_union import summarize


def fixture():
    records = []
    for layer, stride in ((0, 100), (1, 300)):
        for family, scale in (("gate", 1), ("up", 1), ("down", 2)):
            records.append(dict(name=f"blk.{layer}.ffn_{family}_exps.weight.rank0",
                                dims=3, shape=[2560, 640, 512],
                                bytes=512*stride*scale, type=12))
    manifest = dict(sha256="a"*64, records=records)
    provenance = dict(origin="native-target-MTP-verification",
                      manifest_sha256="a"*64, source_commit="b"*40,
                      sessions={"42": dict(case="code-0", category="code")})
    rows = [dict(kind="mtp_verify_routes", version=1, session=42, layer=layer,
                 first=20, n=4, replay=False,
                 experts=[list(range(10)) if layer == 0 else list(range(t*10, t*10+10))
                          for t in range(4)]) for layer in (0, 1)]
    return rows, manifest, provenance


class UnionTests(unittest.TestCase):
    def test_byte_weighted_not_cardinality_average(self):
        rows, manifest, provenance = fixture()
        result = summarize(rows, manifest, provenance, [0, 1])
        self.assertEqual(result["groups"]["all"]["gu_byte_ratio"], 3.25)
        self.assertEqual(result["groups"]["all"]["down_byte_ratio"], 3.25)
        self.assertFalse(result["full_model_scope"])
        self.assertEqual(result["windows"], 1)

    def test_complete_scope_required(self):
        rows, manifest, provenance = fixture()
        with self.assertRaises(ValueError):
            summarize(rows, manifest, provenance)
        with self.assertRaises(ValueError):
            summarize(rows[:1], manifest, provenance, [0, 1])

    def test_replay_not_merged(self):
        rows, manifest, provenance = fixture()
        replay = copy.deepcopy(rows[0])
        replay["replay"] = True
        result = summarize(rows+[replay], manifest, provenance, [0, 1])
        self.assertEqual(result["ignored"]["replay"], 1)

    def test_duplicate_window_rejected(self):
        rows, manifest, provenance = fixture()
        with self.assertRaises(ValueError):
            summarize(rows+rows[:1], manifest, provenance, [0, 1])

    def test_duplicate_expert_rejected(self):
        rows, manifest, provenance = fixture()
        rows[0]["experts"][0][1] = 0
        with self.assertRaises(ValueError):
            summarize(rows, manifest, provenance, [0, 1])

    def test_provenance_and_unknown_sessions_rejected(self):
        rows, manifest, provenance = fixture()
        for change in (dict(origin="synthetic"), dict(manifest_sha256="c"*64),
                       dict(source_commit=""), dict(sessions={})):
            with self.assertRaises(ValueError):
                summarize(rows, manifest, dict(provenance, **change), [0, 1])

    def test_nonuniform_or_partial_banks_rejected(self):
        rows, manifest, provenance = fixture()
        manifest["records"][0]["bytes"] += 1
        with self.assertRaises(ValueError):
            summarize(rows, manifest, provenance, [0, 1])


if __name__ == "__main__":
    unittest.main()
