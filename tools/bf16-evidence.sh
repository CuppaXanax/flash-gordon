#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
test -f offline.complete
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
PYTHONPATH=source/tools python3 - <<'PY'
import json
from pathlib import Path
from mtp_format_audit import load_manifest,PINNED_MANIFEST
def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
m=load_manifest(Path("pack/manifest.fgm"))
assert m["sha256"]==PINNED_MANIFEST
trained=[t for t in m["records"] if t["kind"] in (1,2)]
assert len(trained)==1223
counts={}
for rank in range(8):
    expected={t["name"]:t for t in trained if t["rank"]==rank}
    proof=rows(f"pack/source-{rank}.jsonl")
    assert len(proof)==len(expected)==len({r["tensor"] for r in proof})
    for r in proof:
        assert r["manifest_tensor_sha_exact"] and r["bytes"]==expected[r["tensor"]]["bytes"]
    counts[rank]=len(proof)
ngram=rows("pack/source-ngram.jsonl")
assert len(ngram)==1 and ngram[0]["manifest_tensor_sha_exact"]
assert ngram[0]["bytes"]==28800138240
assert ngram[0]["raw_sha256"]=="dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3"
prepared=rows("prepare.jsonl")
assert prepared[-1]==dict(kind="prepared_complete",cases=80,tokens=1808)
stages=[]
for first,last,rank in (map(int,s.split()) for s in Path("stages.txt").read_text().splitlines()):
    recorded=rows(f"stage-{first}.jsonl")
    summary=recorded[-1]
    assert len(recorded)==81 and {r["case"] for r in recorded[:-1]}==set(range(80))
    assert sum(r["tokens"] for r in recorded[:-1])==1808
    assert summary["first"]==first and summary["last"]==last and summary["rank"]==rank
    assert summary["rounded_boundaries"]==320 and summary["prefix_max_relative"]<=1e-4
    stages.append(summary)
assert len(stages)==8
quality=rows("quality.jsonl")
assert len(quality)==320 and sum(r["final"] for r in quality)==80
Path("source-audit.json").write_text(json.dumps(dict(
    manifest_sha256=m["sha256"],trained_spans=1223,rank_span_counts=counts,
    sealed_tokenizer_sha256="b291b98670183858d414610cfe979e77a67ca53a1b0e977ed568548f8558c820",
    prepared=prepared[-1],stages=stages,quality_observations=len(quality)),indent=2)+"\n")
PY
sha256sum source/tools/bf16-offline source/tools/bench-bf16-hops \
    source/tools/mtp-real-place source/tests/test_bf16_hops > binaries.sha256
find source/shaders source/include source/src -maxdepth 1 -type f \
    \( -name '*.c' -o -name '*.h' -o -name '*.comp' -o -name '*.glsl' \) -print0 \
    | sort -z | xargs -0 sha256sum > numerical-source.sha256
find source/vulkan -name '*.spv' -print0 | sort -z | xargs -0 sha256sum > shaders.sha256
sha256sum pack/rank-*.fgw > rank-pack.pending.sha256
mv rank-pack.pending.sha256 rank-pack.sha256
paths=(source-audit.json binaries.sha256 numerical-source.sha256 shaders.sha256
    rank-pack.sha256 pack/manifest.fgm pack/source-map.json pack/source-*.jsonl
    prompts.fgev pack/cases.bin
    tokenizer-source.json tokenizer-check.log metadata.sha256 stages.txt
    prepared.sha256 prepared-binary.sha256 offline-inputs.sha256 prepare.jsonl prepare.stderr
    stage-*.jsonl stage-*.stderr stage-*.sha256 quality.jsonl quality.stderr
    governor-before.sha256 services-before.txt)
tar -czf evidence-50.tar.gz "${paths[@]}"
sha256sum evidence-50.tar.gz
