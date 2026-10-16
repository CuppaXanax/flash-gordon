#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
test -f fp16.complete
export TMPDIR="$PWD/build"
PYTHONPATH=source/tools python3 - <<'PY'
import json
from pathlib import Path
from mtp_format_audit import load_manifest,PINNED_MANIFEST
m=load_manifest(Path("pack/manifest.fgm"))
assert m["sha256"]==PINNED_MANIFEST
trained=[t for t in m["records"] if t["kind"] in (1,2)]
assert len(trained)==1223
for rank in range(8):
    proof=[json.loads(s) for s in Path(f"pack/source-{rank}.jsonl").read_text().splitlines()]
    expected={t["name"]:t for t in trained if t["rank"]==rank}
    assert len(proof)==len(expected)
    assert all(r["manifest_tensor_sha_exact"] and r["bytes"]==expected[r["tensor"]]["bytes"] for r in proof)
ng=json.loads(Path("pack/source-ngram.jsonl").read_text().strip())
assert ng["raw_sha256"]=="dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3"
screen=json.loads(Path("fp16-screened-policy.json").read_text())
policies=["f32","fp16","fp16-128","bf16","mixed-first-f32","mixed-no-rank0","mixed-screened"]+[f"single-{r}" for r in range(8)]
summary={}
for name in policies:
    quality=[json.loads(s) for s in Path(f"{name}-quality.jsonl").read_text().splitlines()]
    assert len(quality)==320 and sum(r["final"] for r in quality)==80
    assert len(list(Path("logits",name).glob("*.fgl")))==80
    stages=[]
    for p in sorted(Path(".").glob(f"{name}-stage-*.jsonl")):
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        assert len(rows)==81 and {r["case"] for r in rows[:-1]}==set(range(80))
        assert sum(r["tokens"] for r in rows[:-1])==1808
        assert rows[-1]["prefix_max_relative"]<=1e-4
        stages.append(rows[-1])
    expected=(8 if name in ("f32","fp16","fp16-128","bf16","mixed-no-rank0") else
        7 if name=="mixed-first-f32" else
        next((8-i for i,r in enumerate([1,0,2,3,4,5,6,7]) if screen["mask"]&(1<<r)),0)
        if name=="mixed-screened" else 8-[1,0,2,3,4,5,6,7].index(int(name[-1])))
    assert len(stages)==expected
    summary[name]=dict(stages=stages,quality_observations=320)
oracle=[json.loads(s) for s in Path("fp16-real-wire-oracle.jsonl").read_text().splitlines()]
sites=38+screen["mask"].bit_count()
assert len(oracle)==sites and sum(r["rows"] for r in oracle)==sites*320
assert all(r["wire_and_decode_bit_exact"] for r in oracle)
Path("fp16-source-audit.json").write_text(json.dumps(dict(
    manifest_sha256=m["sha256"],trained_spans=1223,ngram_raw_sha256=ng["raw_sha256"],
    policies=summary,screened_policy=screen,optimized_wire_oracle_rows=sites*320),indent=2)+"\n")
PY
sha256sum -c rank-pack.sha256 > fp16-rank-readback.log
sha256sum source/tools/bf16-offline hardware/tools/fp16-wire-oracle \
    hardware/tests/test_bf16_hops source/src/protocol.c hardware/src/protocol.c > fp16-binaries.sha256
find source/shaders source/include source/src -maxdepth 1 -type f \
    \( -name '*.c' -o -name '*.h' -o -name '*.comp' -o -name '*.glsl' \) -print0 \
    | sort -z | xargs -0 sha256sum > fp16-numerical-source.sha256
find source/vulkan -name '*.spv' -print0 | sort -z | xargs -0 sha256sum > fp16-shaders.sha256
tar -czf fp16-evidence-50.tar.gz fp16-source-audit.json fp16-binaries.sha256 \
    fp16-numerical-source.sha256 fp16-shaders.sha256 rank-pack.sha256 fp16-rank-readback.log \
    pack/manifest.fgm pack/source-map.json pack/source-*.jsonl pack/cases.bin \
    prompts.fgev metadata.sha256 fp16-prepared.sha256 fp16-prepare.jsonl fp16-prepare.stderr \
    *-stage-*.jsonl *-stage-*.stderr *-stage-*.sha256 *-quality.jsonl *-quality.stderr \
    *-head.sha256 fp16-real-wire-oracle.jsonl fp16-screened-policy.json governor-before.sha256 services-before.txt \
    source/src/protocol.c source/tools/bf16_offline.c hardware/src/protocol.c
tar -cf - logits | gzip -1 > fp16-logits.tar.gz
sha256sum fp16-evidence-50.tar.gz fp16-logits.tar.gz
touch fp16-evidence.complete
