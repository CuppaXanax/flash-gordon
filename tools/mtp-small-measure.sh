#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp/source
export XDG_CACHE_HOME="$PWD/.build/cache"
mkdir -p .build/small-measure
for seed in 1 2; do
    if [ "$seed" = 1 ]; then order="1 2 3 4"; else order="4 3 2 1"; fi
    for n in $order; do
        name=".build/small-measure/seed${seed}-n${n}"
        cat /sys/class/drm/card*/device/pp_dpm_sclk > "${name}-clock-before.txt" 2>/dev/null || true
        ./tools/bench-decode-fusion --q4 --small-n --batch "$n" --seed "$seed" \
            --warmup "$((256/n))" --iterations "$((64/n))" --windows 3 \
            --dump "${name}.bin" > "${name}.jsonl" 2> "${name}.stderr"
        cat /sys/class/drm/card*/device/pp_dpm_sclk > "${name}-clock-after.txt" 2>/dev/null || true
        echo "finished seed=$seed N=$n"
    done
done
python3 - <<'PY'
import array, json, statistics, pathlib
root=pathlib.Path('.build/small-measure')
result={'host':'192.168.42.51','synthetic':True,'expert_slabs_distinct':True,'runs':[]}
for seed in (1,2):
    baseline=array.array('f'); baseline.frombytes((root/f'seed{seed}-n1.bin').read_bytes())
    base_us=None
    for n in (1,2,3,4):
        rows=[json.loads(x) for x in (root/f'seed{seed}-n{n}.jsonl').read_text().splitlines()]
        samples=[x['wall_us'] for x in rows if x['kind']=='sample']
        mean=statistics.mean(samples)
        if n==1:base_us=mean
        values=array.array('f');values.frombytes((root/f'seed{seed}-n{n}.bin').read_bytes())
        start=(256//n)*n
        begin=max(start,256)
        end=min(start+len(values)//10240,256+len(baseline)//10240)
        bit_equal=0; maximum=0.0
        for t in range(begin,end):
            a=baseline[(t-256)*10240:(t-255)*10240]
            b=values[(t-start)*10240:(t-start+1)*10240]
            bit_equal+=a.tobytes()==b.tobytes()
            maximum=max(maximum,max(abs(x-y) for x,y in zip(a,b)))
        run=dict(seed=seed,n=n,mean_us=mean,p50_us=statistics.median(samples),
                 ratio=mean/base_us,config=rows[0],parity=rows[-1],
                 final_layer_expert_union=next((x for x in rows if x['kind']=='expert_union'),None),
                 compared_tokens=end-begin,bit_equal_tokens=bit_equal,
                 max_abs_hidden_difference=maximum)
        result['runs'].append(run)
        print(json.dumps(run))
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY
tar -cf /home/xander/hc-mtp/small-measure.tar .build/small-measure/*.jsonl \
    .build/small-measure/*.txt .build/small-measure/*.stderr .build/small-measure/summary.json
