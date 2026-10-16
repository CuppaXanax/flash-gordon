#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
df -h /home/xander | tail -n 1
pgrep -af 'source/tools/(bf16-offline|bf16_source.py)' || true
find pack -maxdepth 1 -name 'rank-*.fgw.verified' | wc -l
ls -1 *-head.sha256 2>/dev/null || true
find . -maxdepth 1 -name '*-stage-*.jsonl' -printf '%T@ %p\n' | sort -n | tail -n 3 |
while read -r time path; do printf '%s ' "$path"; tail -n 1 "$path"; done
test ! -f fp16.complete || echo FP16_COMPLETE
python3 - <<'PY'
import json
from pathlib import Path
for proof in sorted(Path(".").glob("*-head.sha256")):
    name=proof.name.removesuffix("-head.sha256")
    rows=[json.loads(s) for s in Path(f"{name}-quality.jsonl").read_text().splitlines()]
    final=[r for r in rows if r["final"]]
    print(json.dumps(dict(policy=name,final_positions=len(final),
        mean_kl=sum(r["kl"] for r in final)/len(final),
        top1=sum(r["teacher_top1"]==r["bf16_top1"] for r in final)/len(final))))
PY
