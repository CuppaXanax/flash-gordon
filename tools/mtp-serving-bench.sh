#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp/source
export TMPDIR="$PWD/.build" XDG_CACHE_HOME="$PWD/.build/cache"
identity=$(sha256sum src/vk.c shaders/fg_small_gate_up.glsl | sha256sum | cut -c1-12)
directory="../serving-bench-$identity"
mkdir -p "$directory"
for n in 1 2 3 4; do
    ./tools/bench-decode-fusion --q4 --small-n --batch "$n" \
        --warmup "$((256/n))" --iterations "$((64/n))" --windows 3 \
        > "$directory/n$n.jsonl" 2> "$directory/n$n.stderr"
done
python3 - "$directory" <<'PY'
import json,pathlib,statistics,sys
directory=pathlib.Path(sys.argv[1]);rows=[]
for n in range(1,5):
    entries=[json.loads(s) for s in (directory/f'n{n}.jsonl').read_text().splitlines()]
    samples=[e['wall_us'] for e in entries if e['kind']=='sample']
    rows.append(dict(n=n,mean_us=statistics.mean(samples),
                     p50_us=statistics.median(samples),samples=len(samples),
                     expert_union=next(e for e in entries if e['kind']=='expert_union')))
for row in rows:row['ratio']=row['mean_us']/rows[0]['mean_us']
(directory/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(directory=str(directory),runs=rows)))
PY
