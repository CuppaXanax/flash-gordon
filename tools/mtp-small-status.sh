#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
if [ -f .build/small-dense-only.jsonl ]; then
    echo "== dense-only"
    tail -12 .build/small-dense-only.jsonl
fi
for f in .build/small-smoke-*.jsonl .build/small-smoke-*.stderr; do
    echo "== $f"
    tail -4 "$f"
done
if [ -f .build/small-isa.stderr ]; then
    python3 - <<'PY'
import json
current=None
for line in open('.build/small-isa.stderr'):
    if line.startswith('VK_PIPELINE'):
        if current: print(json.dumps(current))
        current={'pipeline':line.strip()} if 'fg_small_' in line else None
    elif current and ':' in line:
        key,value=line.strip().split(':',1)
        if key in ('VGPRs','Spilled VGPRs','Scratch size','LDS size','Code size'):
            current[key]=value.strip()
if current: print(json.dumps(current))
PY
fi
