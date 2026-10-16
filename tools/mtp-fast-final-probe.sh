#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
if [ -f .build/fast-clock-sample.json ] && [ ! -d .build/attempt-serial-attention ]; then
    mkdir -p .build/attempt-serial-attention
    cp .build/fast-*.json .build/fast-*.sha256 .build/profile-* .build/attempt-serial-attention/
    tar -czf .build/attempt-serial-attention/source.tar.gz Makefile include src/vk.c src/mtp_head.c \
        shaders tools/bench_mtp_head.c
fi
if [ ! -d .build/attempt-r16 ]; then
    mkdir -p .build/attempt-r16
    cp .build/fast-*.json .build/profile-* .build/attempt-r16/
    tar -czf .build/attempt-r16/source.tar.gz Makefile include src/vk.c src/mtp_head.c \
        shaders tools/bench_mtp_head.c
fi
bash /home/xander/hc-mtp/source/tools/mtp-fast-build.sh
cd /home/xander/hc-mtp/source
bash tools/mtp-expert-quantize.sh q5
bash tools/mtp-fast-probe.sh
for kind in q8 expert-q5; do
    rm -f ".build/fast-${kind}-first.bin" ".build/fast-${kind}-first.bin.trace"
    ./tools/bench-mtp-head --overlay "../native-${kind}.gguf" --embedding-f32 ../embedding.f32 \
        --head-q8 ../shared-head.q8 --replay --head-gpu --dump-first ".build/fast-${kind}-first.bin" \
        --warmup 4 --iterations 4 > ".build/fast-${kind}-full.json"
    cat ".build/fast-${kind}-full.json"
    args=()
    if [ "$kind" != q8 ]; then args=(--ggml-library ../quantizer/build-native/ggml/src/libggml-base.so); fi
    OPENBLAS_NUM_THREADS=4 /home/xander/hc-mtp/.venv/bin/python tools/mtp_head_reference.py \
        --overlay "../native-${kind}.gguf" --head-q8 ../shared-head.q8 \
        --dump ".build/fast-${kind}-first.bin" --output ".build/fast-${kind}-reference.json" "${args[@]}"
done
/home/xander/hc-mtp/.venv/bin/python tools/mtp_head_compare.py --reference .build/fast-q8-first.bin \
    --candidate .build/fast-expert-q5-first.bin --output .build/fast-expert-q5-quality.json
./tools/bench-mtp-head --overlay ../native-expert-q5.gguf --embedding-f32 ../embedding.f32 \
    --replay --compare-reference --warmup 8 --iterations 40 > .build/fast-expert-q5-parity.json
cat .build/fast-expert-q5-parity.json
for shard in 0 1 2 3; do
    ./tools/bench-mtp-head --overlay ../native-expert-q5.gguf --embedding-f32 ../embedding.f32 \
        --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 --head-shard "$shard" \
        --warmup 16 --iterations 32 > ".build/fast-expert-q5-shard-${shard}.json"
    cat ".build/fast-expert-q5-shard-${shard}.json"
done
