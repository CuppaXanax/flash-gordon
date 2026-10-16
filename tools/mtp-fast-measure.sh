#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
cp "$0" .build/fast-measure-recipe.sh
tar -xf ../fast-source.tar tools/mtp-fast-build.sh
bash tools/mtp-fast-build.sh > .build/fast-validation.log 2>&1 \
    || { tail -60 .build/fast-validation.log; exit 1; }
./tools/bench-mtp-head --self-test > .build/fast-self-test.jsonl
for kind in q8 expert-q5; do
    for run in 1 2 3; do
        for shard in 0 1 2 3; do
            ./tools/bench-mtp-head --overlay "../native-${kind}.gguf" --embedding-f32 ../embedding.f32 \
                --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 --head-shard "$shard" \
                --warmup 16 --iterations 32 > ".build/fast-final-${kind}-${run}-${shard}.json"
            cat ".build/fast-final-${kind}-${run}-${shard}.json"
        done
    done
done
for prefix in 512 2048 8192; do
    ./tools/bench-mtp-head --overlay ../native-expert-q5.gguf --embedding-f32 ../embedding.f32 \
        --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 \
        --warmup "$prefix" --iterations 32 > ".build/fast-context-${prefix}.json"
    cat ".build/fast-context-${prefix}.json"
done
device="$(find /sys/class/drm -maxdepth 1 -name 'card[0-9]' | sort | head -1)/device"
cat "$device/pp_dpm_sclk" > .build/fast-clock-before.txt
./tools/bench-mtp-head --overlay ../native-expert-q5.gguf --embedding-f32 ../embedding.f32 \
    --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 \
    --warmup 64 --iterations 512 > .build/fast-clock-sample.json &
pid=$!
while [ -e "/proc/$pid/status" ]; do
    { date +%s.%N; cat "$device/pp_dpm_sclk"; } >> .build/fast-clock-observations.txt
    sleep 0.01
done
wait "$pid"
cat "$device/pp_dpm_sclk" > .build/fast-clock-after.txt
sha256sum tools/bench-mtp-head flash-gordon vulkan/fg_mtp*.spv \
    vulkan/fg_dense_q8_0_cooked_r8.spv > .build/fast-binaries.sha256
sha256sum ../fast-source.tar > .build/fast-archive.sha256
sha256sum ../native-q8.gguf ../native-q4.gguf ../native-expert-q4.gguf \
    ../native-expert-q5.gguf ../shared-head.q8 ../embedding.f32 \
    ../quantizer/build-native/ggml/src/libggml-base.so ../ggml-source.tar \
    > .build/fast-inputs.sha256
find include src shaders tests tools -type f \
    \( -name '*.c' -o -name '*.h' -o -name '*.comp' -o -name '*.glsl' \
    -o -name '*.py' -o -name '*.sh' -o -name '*.ps1' \) -print0 |
    sort -z | xargs -0 sha256sum > .build/fast-sources.sha256
sha256sum Makefile glslc >> .build/fast-sources.sha256
{ uname -a; /home/xander/hc-mtp/.venv/bin/python -c \
    'import sys,numpy;print(sys.version);print("numpy",numpy.__version__)'; } > .build/fast-platform.txt
cp ../native-source.json .build/fast-source.json
cp ../shared-source.json .build/fast-shared-source.json
for kind in q4 expert-q4 expert-q5; do
    cp "../native-${kind}.json" ".build/fast-${kind}-export.json"
done
tar -czf /home/xander/hc-mtp/fast-evidence.tar.gz \
    .build/fast-* .build/profile-* .build/attempt-r16 .build/attempt-serial-attention
sha256sum /home/xander/hc-mtp/fast-evidence.tar.gz
