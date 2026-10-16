#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp/source
export TMPDIR="$PWD/.build" XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 tools/bench-decode-fusion > ../verify-profile-build.log 2>&1
for n in 2 4; do
    ./tools/bench-decode-fusion --q4 --small-n --batch "$n" \
        --warmup "$((256/n))" --iterations 1 --windows 1 --profile \
        > "../verify-profile-$n.jsonl" 2> "../verify-profile-$n.stderr"
done
bash tools/mtp-small-isa.sh
python3 - <<'PY'
import json,pathlib
for n in (2,4):
    rows=[json.loads(s) for s in pathlib.Path(f'../verify-profile-{n}.jsonl').read_text().splitlines()]
    print(json.dumps(dict(n=n,kernels=[r for r in rows if r['kind']=='kernel'
        and ('fg_small_q4' in r['name'] or 'fg_small_gate' in r['name']
             or 'fg_small_down' in r['name'])])))
PY
