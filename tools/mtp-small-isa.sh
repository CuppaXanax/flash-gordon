#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp/source
export XDG_CACHE_HOME="$PWD/.build/cache"
RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true \
    ./tools/bench-decode-fusion --q4 --small-n --batch 4 \
    --warmup 2 --iterations 1 --windows 1 \
    > .build/small-isa.jsonl 2> .build/small-isa.stderr
python3 - <<'PY'
import json
records=[]; current=None
for line in open('.build/small-isa.stderr'):
    if line.startswith('VK_PIPELINE'):
        if current: records.append(current)
        current={'pipeline':line.strip()}
    elif current and ':' in line:
        key,value=line.strip().split(':',1)
        if key in ('VGPRs','Spilled VGPRs','Scratch size','LDS size','Code size','Wave size'):
            current[key]=value.strip()
if current: records.append(current)
for record in records:
    if 'fg_small_' in record['pipeline']: print(json.dumps(record))
with open('.build/small-isa-summary.json','w') as output:
    json.dump(records,output,indent=2); output.write('\n')
PY
