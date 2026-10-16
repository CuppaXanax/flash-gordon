#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
export TMPDIR="$PWD/.build" XDG_CACHE_HOME="$PWD/.build/cache"
directory=.build/verify-ab
sha256sum --quiet -c "$directory/sha256.txt"
make -j8 tools/bench-decode-fusion > ../verify-repeat-build.log 2>&1
for seed in 7 17; do
    for n in 1 2 3 4; do
        for round in a1 b a2; do
            if [[ "$round" == b ]]; then binary=candidate; else binary=baseline; fi
            FG_SHADER_DIR="$PWD/$directory/$binary-vulkan" "$directory/$binary" \
                --q4 --small-n --batch "$n" --seed "$seed" \
                --warmup "$((256/n))" --iterations "$((64/n))" --windows 3 \
                > "$directory/seed$seed-$round-n$n.jsonl" \
                2> "$directory/seed$seed-$round-n$n.stderr"
        done
    done
done
python3 - "$directory" <<'PY'
import json,pathlib,statistics,sys
d=pathlib.Path(sys.argv[1]);result=[]
for seed in (0,7,17):
    means={}
    for n in range(1,5):
        r=dict(seed=seed,n=n);parity={}
        for label in ('a1','b','a2'):
            name=f'{label}-n{n}' if not seed else f'seed{seed}-{label}-n{n}'
            entries=list(map(json.loads,(d/f'{name}.jsonl').read_text().splitlines()))
            r[label]=statistics.mean(e['wall_us'] for e in entries if e['kind']=='sample')
            parity[label]=next(e for e in entries if e['kind']=='parity')
        assert parity['a1']==parity['b']==parity['a2'],(seed,n,parity)
        r['output_state_hashes_equal']=True
        r['candidate_vs_bracket']=r['b']/statistics.mean(r[x] for x in ('a1','a2'))
        means[n]=r
    for n,r in means.items():
        r['candidate_ratio']=r['b']/means[1]['b']
        result.append(r)
(d/'repeat-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(kind='three_seed_brackets',runs=result)))
PY
bash tools/mtp-verify-profile.sh
