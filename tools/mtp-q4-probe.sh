#!/usr/bin/env bash
set -euo pipefail
bash /home/xander/hc-mtp/source/tools/mtp-fast-build.sh
cd /home/xander/hc-mtp/source
test -s ../native-q4.json
rm -f .build/fast-q8-first.bin .build/fast-q4-first.bin .build/fast-q8-first.bin.trace .build/fast-q4-first.bin.trace
for kind in q8 q4; do
    ./tools/bench-mtp-head --overlay "../native-${kind}.gguf" --embedding-f32 ../embedding.f32 \
        --head-q8 ../shared-head.q8 --replay --head-gpu --dump-first ".build/fast-${kind}-first.bin" \
        --warmup 4 --iterations 4 > ".build/fast-${kind}-full.json"
    cat ".build/fast-${kind}-full.json"
    args=()
    if [ "$kind" = q4 ]; then args=(--ggml-library ../quantizer/build-native/ggml/src/libggml-base.so); fi
    OPENBLAS_NUM_THREADS=4 ../.venv/bin/python tools/mtp_head_reference.py \
        --overlay "../native-${kind}.gguf" --head-q8 ../shared-head.q8 \
        --dump ".build/fast-${kind}-first.bin" --output ".build/fast-${kind}-reference.json" "${args[@]}"
done
../.venv/bin/python tools/mtp_head_compare.py --reference .build/fast-q8-first.bin \
    --candidate .build/fast-q4-first.bin --output .build/fast-q4-quality.json
./tools/bench-mtp-head --overlay ../native-q4.gguf --embedding-f32 ../embedding.f32 \
    --replay --compare-reference --warmup 4 --iterations 4 > .build/fast-q4-parity.json
cat .build/fast-q4-parity.json
for shard in 0 1 2 3; do
    ./tools/bench-mtp-head --overlay ../native-q4.gguf --embedding-f32 ../embedding.f32 \
        --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 --head-shard "$shard" \
        --warmup 16 --iterations 32 > ".build/fast-q4-shard-${shard}.json"
    cat ".build/fast-q4-shard-${shard}.json"
done
