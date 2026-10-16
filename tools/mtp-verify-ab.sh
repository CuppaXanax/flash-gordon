#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
mkdir -p source .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
tar -xf source.tar -C source
cd source
directory=.build/verify-ab
mkdir -p "$directory/baseline-source" "$directory/baseline-vulkan" \
    "$directory/candidate-vulkan"
tar -xf mtp-verify-base.tar -C "$directory/baseline-source"
make -j8 tools/bench-decode-fusion > ../verify-ab-build.log 2>&1 \
    || { tail -70 ../verify-ab-build.log; exit 1; }
./tools/bench-decode-fusion --q4 --small-n --batch 2 --small-test --rollback-test \
    --warmup 1 --iterations 1 --windows 1 > "$directory/component-gates.jsonl" \
    2> "$directory/component-gates.stderr"
cp tools/bench-decode-fusion "$directory/candidate"
cp vulkan/*.spv "$directory/baseline-vulkan/"
cp vulkan/*.spv "$directory/candidate-vulkan/"
cp src/vk.c "$directory/candidate-vk.c"
cp "$directory/baseline-source/src/vk.c" src/vk.c
trap 'cp "$directory/candidate-vk.c" src/vk.c' EXIT
cat >> src/vk.c <<'C'

fg_status fg_vk_small_n_tune(fg_vk_context *c,uint32_t wg,uint32_t rows,fg_error *err){
    (void)c;(void)wg;(void)rows;
    fg_error_set(err,FG_ERR_ARGUMENT,"baseline geometry is immutable");
    return FG_ERR_ARGUMENT;
}
C
make -j8 tools/bench-decode-fusion >> ../verify-ab-build.log 2>&1 \
    || { tail -70 ../verify-ab-build.log; exit 1; }
cp tools/bench-decode-fusion "$directory/baseline"
cp "$directory/candidate-vk.c" src/vk.c
trap - EXIT
for shader in "$directory"/baseline-source/shaders/fg_small*.comp \
    "$directory"/baseline-source/shaders/fg_gdn_small.comp; do
    name=$(basename "$shader" .comp)
    sh ./glslc -O --target-env=vulkan1.2 \
        -o "$directory/baseline-vulkan/$name.spv" "$shader" \
        >> ../verify-ab-build.log 2>&1
done
sha256sum "$directory/baseline" "$directory/candidate" \
    "$directory"/{baseline,candidate}-vulkan/fg_small*.spv \
    "$directory"/{baseline,candidate}-vulkan/fg_gdn_small.spv > "$directory/sha256.txt"
for n in 1 2 3 4; do
    for round in a1 b a2 c; do
        tune=()
        if [[ "$round" == a* ]]; then
            binary=baseline
            shader=baseline-vulkan
        else
            binary=candidate
            shader=candidate-vulkan
            if [[ "$round" == c ]]; then
                tune=(--small-dense-wg 32 --small-dense-rows 8)
            fi
        fi
        FG_SHADER_DIR="$PWD/$directory/$shader" \
            "$directory/$binary" --q4 --small-n --batch "$n" "${tune[@]}" \
            --warmup "$((256/n))" --iterations "$((64/n))" --windows 3 \
            > "$directory/$round-n$n.jsonl" 2> "$directory/$round-n$n.stderr"
    done
done
python3 - "$directory" <<'PY'
import json,pathlib,statistics,sys
d=pathlib.Path(sys.argv[1])
rows=[]
for n in range(1,5):
    r=dict(n=n)
    parity={}
    for label in ('a1','b','a2','c'):
        entries=list(map(json.loads,(d/f'{label}-n{n}.jsonl').read_text().splitlines()))
        samples=[e['wall_us'] for e in entries if e['kind']=='sample']
        parity[label]=next(e for e in entries if e['kind']=='parity')
        r[label]=dict(mean_us=statistics.mean(samples),p50_us=statistics.median(samples))
    assert parity['a1']==parity['b']==parity['a2'],(n,parity)
    r['baseline_candidate_hashes_equal']=True
    r['diagnostic_hash_equal']=parity['c']==parity['a1']
    r['candidate_vs_bracket']=r['b']['mean_us']/statistics.mean(
        r[x]['mean_us'] for x in ('a1','a2'))
    rows.append(r)
for r in rows:
    for label in ('a1','b','a2','c'):
        r[label]['ratio']=r[label]['mean_us']/rows[0][label]['mean_us']
(d/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(kind='ab_replay',runs=rows)))
PY
cp "$directory/candidate" tools/bench-decode-fusion
