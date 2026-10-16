#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
tar -xf source.tar -C source
cd source
make -j8 flash-gordon tools/bench-decode-fusion tests/test_mtp tests/test_sampler \
    tests/test_decode_batch > .build/validate-build.log 2>&1 || {
    tail -80 .build/validate-build.log
    exit 1
}
bash tools/check-flags.sh
./tests/test_mtp
./tests/test_sampler
./tests/test_decode_batch
for warmup in 5 6 7 8; do
    ./tools/bench-decode-fusion --q4 --batch 1 --warmup "$warmup" \
        --iterations 2 --windows 1 --rollback-test \
        > ".build/rollback-${warmup}.jsonl" 2> ".build/rollback-${warmup}.stderr"
done
python3 - <<'PY'
import json,pathlib
result={'host':'192.168.42.51','unit_mtp':True,'unit_sampler':True,
        'unit_decode_batch':True,'flags':True,'build_werror':True,'rollback':[]}
for warmup in range(5,9):
    rows=[json.loads(x) for x in pathlib.Path(f'.build/rollback-{warmup}.jsonl').read_text().splitlines()]
    for r in rows:
        if r['kind']=='rollback':
            r['initial_frontier']=warmup+2
            result['rollback'].append(r)
assert len(result['rollback'])==16
assert all(r['gdn_and_output_bit_exact'] for r in result['rollback'])
pathlib.Path('.build/validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
