#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
test .build/small-measure/summary.json -nt .build/small-build.log
./tests/test_mtp > .build/small-host.log
./tests/test_sampler >> .build/small-host.log
./tests/test_decode_batch >> .build/small-host.log
bash tools/check-flags.sh >> .build/small-host.log
sha256sum flash-gordon tools/bench-decode-fusion > .build/small-binaries.sha256
sha256sum vulkan/fg_small*.spv vulkan/fg_gdn_small.spv \
    vulkan/fg_gdn_recurrent_algebraic.spv > .build/small-shaders.sha256
python3 - <<'PY'
import json, pathlib, statistics
root=pathlib.Path('.build')
result=json.loads((root/'small-measure/summary.json').read_text())
smoke=[json.loads(x) for x in (root/'small-smoke-1.jsonl').read_text().splitlines()]
def cases(kind,field,expected):
    rows=[x for x in smoke if x['kind']==kind]
    assert len(rows)==expected and all(x[field] for x in rows),(kind,rows)
    return len(rows)
validation={
    'dense_byte_exact_cases':cases('dense_small_test','bit_exact',90),
    'union_scheduler_cases':cases('small_union_test','exact',12),
    'block_output_byte_exact_cases':cases('small_block_test','output_bit_exact',3),
    'block_state_byte_exact_cases':cases('small_block_test','state_bit_exact',3),
    'causal_prefix_cases':cases('small_causal_test','prefix_bit_exact',1),
    'injected_error_cleanup_cases':cases('small_error_cleanup_test','clean',1),
}
boundary=[]
for offset in range(4):
    rows=[json.loads(x) for x in (root/f'small-boundary-{offset}.jsonl').read_text().splitlines()]
    rows=[x for x in rows if x['kind']=='rollback']
    assert len(rows)==4 and all(x['gdn_and_output_bit_exact'] for x in rows)
    boundary.extend(rows)
validation['page_boundary_prefix_rollback_cases']=len(boundary)
runs=result['runs']
assert len(runs)==8
for run in runs:
    assert run['compared_tokens']==run['bit_equal_tokens'] and run['max_abs_hidden_difference']==0
result['validation']=validation
result['mean_cost_ratios']=[statistics.mean(x['ratio'] for x in runs if x['n']==n) for n in range(1,5)]
result['mean_block_us']=[statistics.mean(x['mean_us'] for x in runs if x['n']==n) for n in range(1,5)]
result['aligned_nonbaseline_byte_exact_rows']=sum(x['bit_equal_tokens'] for x in runs if x['n']!=1)
result['isolated_dense_costs']=[x for x in smoke if x['kind']=='dense_cost']
result['acceptance']={'fg_measured':False,'reference':'Strata 9259cad docs/DETAILS.md:743',
    'estimated_committed_tokens_per_pass':[2.4,3.2],'draft_cap':3}
result['whole_ring_measured']=False
result['target_met']={'n2':result['mean_cost_ratios'][1]<=1.3,
    'n4':result['mean_cost_ratios'][3]<=1.8}
(root/'small-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('mean_cost_ratios','mean_block_us',
    'aligned_nonbaseline_byte_exact_rows','validation','target_met')}))
isa=json.loads((root/'small-isa-summary.json').read_text())
for row in isa:
    if any(name in row['pipeline'] for name in ('fg_small_gate_up','fg_small_down')):
        print(json.dumps(row))
PY
tar -czf /home/xander/hc-mtp/small-evidence.tar.gz \
    .build/small-result.json .build/small-measure/*.jsonl \
    .build/small-measure/*.txt .build/small-measure/*.stderr \
    .build/small-measure/summary.json \
    .build/small-smoke-*.jsonl .build/small-smoke-*.stderr \
    .build/small-boundary-*.jsonl .build/small-boundary-*.stderr \
    .build/small-build.log .build/small-host.log .build/small-binaries.sha256 \
    .build/small-shaders.sha256 .build/small-isa.jsonl \
    .build/small-isa.stderr .build/small-isa-summary.json
