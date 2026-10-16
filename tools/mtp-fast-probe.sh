#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
mkdir -p .build
test -s ../native-source.json
test -s ../shared-source.json
for mode in baseline replay; do
    args=()
    if [ "$mode" = replay ]; then args=(--replay); fi
    ./tools/bench-mtp-head --overlay ../native-q8.gguf --embedding-f32 ../embedding.f32 \
        --warmup 16 --iterations 32 "${args[@]}" > ".build/fast-${mode}.json"
    ./tools/bench-mtp-head --overlay ../native-q8.gguf --embedding-f32 ../embedding.f32 \
        --warmup 4 --iterations 2 --profile "${args[@]}" > ".build/profile-${mode}.json" \
        2> ".build/profile-${mode}.log"
    cat ".build/fast-${mode}.json" ".build/profile-${mode}.log"
done
./tools/bench-mtp-head --overlay ../native-q8.gguf --embedding-f32 ../embedding.f32 \
    --replay --compare-reference --warmup 8 --iterations 40 > .build/fast-parity.json
cat .build/fast-parity.json
for shard in 0 1 2 3; do
    ./tools/bench-mtp-head --overlay ../native-q8.gguf --embedding-f32 ../embedding.f32 \
        --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 --head-shard "$shard" \
        --warmup 16 --iterations 32 > ".build/fast-shard-${shard}.json"
    cat ".build/fast-shard-${shard}.json"
done
