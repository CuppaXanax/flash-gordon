#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p source
tar -xf source.tar -C source
cd source
mkdir -p .build
make -j8 tools/bench-decode-fusion tests/test_mtp tests/test_sampler > .build/build.log 2>&1 || {
    tail -80 .build/build.log
    exit 1
}
./tests/test_mtp
./tests/test_sampler
./tools/bench-decode-fusion --q4 --batch 1 --warmup 5 --iterations 2 --windows 1 \
    --rollback-test > .build/smoke-n1.jsonl
./tools/bench-decode-fusion --q4 --batch 2 --warmup 4 --iterations 2 --windows 1 > .build/smoke-n2.jsonl
python3 - <<'PY'
import json
for n in (1,2):
    rows=[json.loads(x) for x in open(f'.build/smoke-n{n}.jsonl')]
    print(n,rows[-1])
    for row in rows:
        if row['kind']=='rollback': print(row)
PY
