#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
mkdir -p source .cache evidence
sha256sum /etc/cyan-skillfish-governor-smu/config.toml > evidence/governor.sha256
finish() {
    local rc=$?
    cd /home/xander/hc-mtp
    sha256sum -c evidence/governor.sha256 > evidence/governor-check.txt || rc=1
    printf '%s\n' "$rc" > evidence/exit-code.txt
    tar -czf format-evidence.tar.gz evidence
    exit "$rc"
}
trap finish EXIT
sudo -n prlimit --pid "$$" --memlock=268435456:268435456
tar -xf source.tar -C source
export TMPDIR="$PWD/.cache" XDG_CACHE_HOME="$PWD/.cache"
cd source
mkdir -p .build
make -j8 all test-mtp test-multi-session tools/bench-mtp-head \
    tools/bench-decode-fusion tests/test_fg_vk tools/mtp-real-cook \
    > ../evidence/build.log 2>&1 || { tail -60 ../evidence/build.log; exit 1; }
bash tools/check-flags.sh > ../evidence/flags.txt
export FG_SHADER_DIR="$PWD/vulkan" FG_ROOFLINE=2047 FG_ROOF_DENSE=19
export FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024 FG_ROOF_HC=96
export FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 FG_NGRAM_LOCAL_CACHE=1
export FG_DECODE_SUBMIT_GROUP=2
env | LC_ALL=C sort | grep '^FG_' > ../evidence/production-recipe.txt
python3 tools/mtp_format_audit.py --manifest ../manifest.fgm \
    --overlay ../native-q8.gguf --output ../evidence/inventory.json
./tests/test_mtp_embedding .build/format-embedding --gpu \
    > ../evidence/embedding.txt 2>&1
python3 - <<'PY' > ../evidence/cooker-negative.txt 2>&1
from pathlib import Path
import hashlib,subprocess
raw=Path('../layer2-down.raw')
bad=Path('../bad-down.cooked')
command=['./tools/mtp-real-cook','../manifest.fgm',
         'blk.2.ffn_down_exps.weight.rank1',str(raw),str(bad)]
with raw.open('r+b') as file:
    original=file.read(1)
    try:
        file.seek(0);file.write(bytes([original[0]^1]));file.flush()
        result=subprocess.run(command,check=False)
        assert result.returncode!=0 and not bad.exists()
    finally:
        file.seek(0);file.write(original);file.flush()
existing=Path('../layer2-down.cooked')
with existing.open('rb') as file:
    before=hashlib.file_digest(file,'sha256').hexdigest()
command[-1]=str(existing)
assert subprocess.run(command,check=False).returncode!=0
with existing.open('rb') as file:
    assert hashlib.file_digest(file,'sha256').hexdigest()==before
print('Corruption rejected/removed; existing qualified slab preserved: PASS')
PY
for wide in 0 1; do
    for mask in 0 330; do
        FG_EXPERT_Q8_WIDE=$wide FG_SMALL_KERNELS=$mask \
        FG_TEST_Q8_DOWN_TENSOR="$PWD/../layer2-down.cooked" \
        DS4_REMOTE_TEST_FILTER=small_q8_down_native ./tests/test_fg_vk \
            > "../evidence/real-q8-w$wide-m$mask.txt" 2>&1
    done
done
export FG_EXPERT_Q8_WIDE=1
python3 - <<'PY' > ../evidence/dense-shapes.tsv
from tools.mtp_format_audit import load_manifest
manifest=load_manifest('../manifest.fgm')
shapes=set()
for record in manifest['records']:
    if record['kind']!=1:
        continue
    name=record['name']
    if name=='token_embd.weight':
        continue
    if record['dims']==1 and name.endswith('ffn_gate_inp_shexp.weight'):
        shapes.add((0,2560,1))
    if record['dims']!=2:
        continue
    kind=record['type']
    if kind==30 and name.endswith('ffn_gate_inp.weight'):
        kind=31
    width,rows=record['shape']
    if name=='output.weight':
        rows//=4
    if kind in (0,8,12,13,20,30,31) and width%32==0:
        shapes.add((kind,width,rows))
for shape in sorted(shapes):
    print(*shape,sep='\t')
PY
for mask in 0 330; do
    export FG_SMALL_KERNELS=$mask
    for oracle in ple_decode ple_conv_add_parity ple_prefill_scan ple_prefill_t1_compat \
        group_norm hc_finalize hc_inject_partial gdn_prefill_decode_compat \
        qsa_indexer qsa_quant_and_bf16 qsa_resident_causal_batch_attention \
        qsa_resident_t1_compat ngram_direct_lookup ngram_resident ngram_prefill_lookup; do
        DS4_REMOTE_TEST_FILTER="$oracle" ./tests/test_fg_vk \
            > "../evidence/oracle-m$mask-$oracle.txt" 2>&1
    done
    DS4_REMOTE_TEST_FILTER=expert_small_component ./tests/test_fg_vk \
        > "../evidence/expert-component-m$mask.txt" 2>&1
    while read -r type width rows; do
        ./tools/bench-decode-fusion --dense-only --allow-reassociation \
            --dense-type "$type" --dense-shape "$width" "$rows" \
            > "../evidence/dense-m$mask-t$type-i$width-o$rows.jsonl" \
            2> "../evidence/dense-m$mask-t$type-i$width-o$rows.stderr"
    done < ../evidence/dense-shapes.tsv
    ./tools/bench-decode-fusion --q4 --mixed-experts --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../evidence/mixed-m$mask.jsonl" 2> "../evidence/mixed-m$mask.stderr"
    ./tools/bench-mtp-head --overlay ../native-q8.gguf --session-test \
        > "../evidence/native-m$mask.jsonl" 2> "../evidence/native-m$mask.stderr"
done
sha256sum flash-gordon > ../evidence/binary.sha256
find vulkan -type f -name '*.spv' -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum > ../evidence/shaders.sha256
printf 'All real-format qualification gates passed\n'
