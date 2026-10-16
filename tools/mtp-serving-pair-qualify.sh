#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp/source
export TMPDIR="$PWD/.build" XDG_CACHE_HOME="$PWD/.build/cache"
cp vulkan/fg_small_gate_up_four.spv .build/mtp-four-candidate.spv
trap 'cp .build/mtp-four-candidate.spv vulkan/fg_small_gate_up_four.spv' EXIT
for phase in A B A2; do
    if [[ "$phase" == B ]]; then
        cp .build/mtp-four-candidate.spv vulkan/fg_small_gate_up_four.spv
    else
        cp vulkan/fg_small_gate_up_pairs.spv vulkan/fg_small_gate_up_four.spv
    fi
    directory="../serving-pair-$phase"
    mkdir -p "$directory"
    sha256sum vulkan/fg_small_gate_up_four.spv > "$directory/shader.sha256"
    for n in 1 2 3 4; do
        ./tools/bench-decode-fusion --q4 --small-n --batch "$n" \
            --warmup "$((256/n))" --iterations "$((64/n))" --windows 3 \
            > "$directory/n$n.jsonl" 2> "$directory/n$n.stderr"
    done
done
python3 - <<'PY'
import json,pathlib,statistics
phases={};hashes={}
for phase in ['A','B','A2']:
    rows=[]
    for n in range(1,5):
        entries=[json.loads(s) for s in pathlib.Path(f'../serving-pair-{phase}/n{n}.jsonl').read_text().splitlines()]
        samples=[e['wall_us'] for e in entries if e['kind']=='sample']
        rows.append(dict(n=n,mean_us=statistics.mean(samples),p50_us=statistics.median(samples)))
        digest=next(e for e in entries if e['kind']=='parity')
        assert n not in hashes or hashes[n]==digest
        hashes[n]=digest
    for row in rows:row['ratio']=row['mean_us']/rows[0]['mean_us']
    phases[phase]=rows
result=dict(phases=phases,all_output_and_state_hashes_equal=True)
result['candidate_vs_bracketed_baseline']=[
    phases['B'][i]['mean_us']/statistics.mean([phases['A'][i]['mean_us'],phases['A2'][i]['mean_us']])
    for i in range(4)]
pathlib.Path('../serving-pair-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
