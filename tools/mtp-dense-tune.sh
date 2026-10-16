#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
mkdir -p source .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
tar -xf source.tar -C source
cd source
mkdir -p .build/dense-tune
make -j8 tools/bench-decode-fusion > ../dense-tune-build.log 2>&1 \
    || { tail -70 ../dense-tune-build.log; exit 1; }
for wg in 32 64 128 256; do
    for rows in 4 8; do
        for shape in '2560 6144' '2560 10240' '6144 2560' '10240 320'; do
            read -r input output <<< "$shape"
            name="w${wg}r${rows}-${input}x${output}"
            ./tools/bench-decode-fusion --dense-only --dense-type 12 \
                --allow-reassociation \
                --dense-shape "$input" "$output" \
                --small-dense-wg "$wg" --small-dense-rows "$rows" \
                > ".build/dense-tune/$name.jsonl" 2> ".build/dense-tune/$name.stderr"
        done
    done
done
python3 - <<'PY'
import json,pathlib
rows=[]
for path in sorted(pathlib.Path('.build/dense-tune').glob('*.jsonl')):
    entries=[json.loads(s) for s in path.read_text().splitlines()]
    for e in entries:
        if e['kind']=='dense_cost' and e['type']==12:
            rows.append(dict(candidate=path.stem,**e))
pathlib.Path('../dense-tune-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for shape in ('2560x6144','2560x10240','6144x2560','10240x320'):
    for n in (2,3,4):
        choices=sorted((r for r in rows if r['candidate'].endswith(shape) and r['n']==n),
                       key=lambda r:r['us'])
        print(json.dumps(dict(shape=shape,n=n,best=choices[:3])))
PY
