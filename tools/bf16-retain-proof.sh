#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
exec 9>retention.lock
flock 9
if [[ -f rank-pack.sha256 ]]; then
    cat rank-pack.sha256
    exit 0
fi
PYTHONPATH=source/tools python3 - <<'PY' > retention-available.txt
import json
from pathlib import Path
from mtp_format_audit import load_manifest,PINNED_MANIFEST
m=load_manifest(Path("pack/manifest.fgm"))
assert m["sha256"]==PINNED_MANIFEST
for path in sorted(Path(".").glob("stage-*.jsonl")):
    lines=path.read_text().splitlines()
    if not lines:
        continue
    try:
        summary=json.loads(lines[-1])
    except json.JSONDecodeError:
        continue
    if summary["kind"]!="stage_complete":
        continue
    rank=summary["rank"]
    expected={t["name"]:t for t in m["records"] if t["kind"] in (1,2) and t["rank"]==rank}
    proof=[json.loads(s) for s in Path(f"pack/source-{rank}.jsonl").read_text().splitlines()]
    assert len(proof)==len(expected)==len({r["tensor"] for r in proof})
    assert all(r["manifest_tensor_sha_exact"] and r["bytes"]==expected[r["tensor"]]["bytes"] for r in proof)
    print(f"pack/rank-{rank:02d}.fgw")
PY
touch retention-available.sha256
while read -r path; do
    if ! grep -Fq "  $path" retention-available.sha256; then
        sha256sum "$path" >> retention-available.sha256
    fi
done < retention-available.txt
cat retention-available.sha256
