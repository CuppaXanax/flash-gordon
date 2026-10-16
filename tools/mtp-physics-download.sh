#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
cd /home/xander/hc-mtp
{
    date -u
    uname -a
    sha256sum source/src/vk.c source/shaders/fg_small*.glsl \
        source/shaders/fg_small*.comp source/shaders/fg_dense_soa_impl.glsl \
        source/shaders/fg_gdn_small.comp \
        source/tools/bench_decode_fusion.c source/tools/bench-decode-fusion
} > physics-receipt.txt
files=(physics-receipt.txt *physics*.json* *physics*.stderr \
    *verify*.jsonl *verify*.stderr *verify*.log state-cache*.jsonl \
    state-cache*.stderr state-cache*.log \
    rowpair*.jsonl rowpair*.stderr rowpair*.log serving-*.jsonl serving-*.stderr \
    serving-build.log \
    dense-tune-summary.json dense-tune-build.log serving-bench-* \
    source/.build/dense-tune source/.build/verify-ab \
    source/.build/small-isa-summary.json source/.build/small-isa.stderr \
    source/.build/small-isa.jsonl source/src/vk.c \
    source/shaders/fg_small*.glsl source/shaders/fg_small*.comp \
    source/shaders/fg_dense_soa_impl.glsl source/shaders/fg_gdn_small.comp \
    source/tools/bench_decode_fusion.c)
present=()
for file in "${files[@]}"; do
    [[ ! -e "$file" ]] || present+=("$file")
done
tar -czf physics-current-evidence.tar.gz "${present[@]}"
sha256sum physics-current-evidence.tar.gz
