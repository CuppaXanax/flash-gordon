#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
export TMPDIR="$PWD/.build" XDG_CACHE_HOME="$PWD/.build/cache"
make tools/bench-decode-fusion > ../state-cache-build.log 2>&1
./tools/bench-decode-fusion --q4 --small-n --batch 2 \
    --small-test --rollback-test --warmup 1 --iterations 1 --windows 1 \
    > ../state-cache-gates.jsonl 2> ../state-cache-gates.stderr \
    || { tail -15 ../state-cache-gates.jsonl; cat ../state-cache-gates.stderr; exit 1; }
grep -E 'small_block_test|small_causal_test|rollback' ../state-cache-gates.jsonl
for n in 2 4; do
    ./tools/bench-decode-fusion --q4 --small-n --batch "$n" \
        --warmup "$((256/n))" --iterations 1 --windows 1 --profile \
        > "../state-cache-profile-$n.jsonl" 2> "../state-cache-profile-$n.stderr"
done
python3 - <<'PY'
import json,pathlib
for n in (2,4):
    r=[json.loads(s) for s in pathlib.Path(f'../state-cache-profile-{n}.jsonl').read_text().splitlines()]
    print(json.dumps(dict(n=n,kernels=[x for x in r if x['kind']=='kernel'
          and ('recurrent' in x['name'] or 'conv' in x['name'] or 'gdn_small' in x['name'])])))
PY
