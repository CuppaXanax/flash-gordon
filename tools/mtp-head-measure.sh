#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
test -s native-source.json
test -s shared-source.json
cd source
mkdir -p .build
device="$(find /sys/class/drm -maxdepth 1 -name 'card[0-9]' | sort | head -1)/device"
cat "$device/pp_dpm_sclk" > .build/native-clock-before.txt
./tools/bench-mtp-head --self-test > .build/native-self-test.jsonl
rm -f .build/native-first.bin .build/native-first.bin.trace
for run in 1 2 3; do
    extra=()
    if [ "$run" = 1 ]; then extra=(--dump-first .build/native-first.bin); fi
    ./tools/bench-mtp-head --overlay ../native-q8.gguf \
        --head-q8 ../shared-head.q8 --embedding-f32 ../embedding.f32 \
        --warmup 16 --iterations 32 "${extra[@]}" \
        > ".build/native-${run}.jsonl" 2> ".build/native-${run}.stderr"
    cat ".build/native-${run}.jsonl"
done
OPENBLAS_NUM_THREADS=4 /home/xander/hc-mtp/.venv/bin/python tools/mtp_head_reference.py \
    --overlay ../native-q8.gguf --head-q8 ../shared-head.q8 \
    --dump .build/native-first.bin --output .build/native-reference.json
cat "$device/pp_dpm_sclk" > .build/native-clock-after.txt
sha256sum tools/bench-mtp-head flash-gordon vulkan/fg_mtp_attention.spv \
    vulkan/fg_moe_decode_gate_up.spv vulkan/fg_moe_kquant_cooked.spv \
    > .build/native-binaries.sha256
sha256sum ../source.tar > .build/native-archive.sha256
sha256sum ../native-q8.gguf ../shared-head.q8 ../embedding.f32 > .build/native-inputs.sha256
find include src shaders tests tools -type f \
    \( -name '*.c' -o -name '*.h' -o -name '*.comp' -o -name '*.glsl' \
    -o -name '*.py' -o -name '*.sh' -o -name '*.ps1' \) -print0 |
    sort -z | xargs -0 sha256sum > .build/native-sources.sha256
{ uname -a; /home/xander/hc-mtp/.venv/bin/python -c \
    'import sys,numpy;print(sys.version);print("numpy",numpy.__version__)'; } \
    > .build/native-platform.txt
cp ../native-source.json .build/native-source.json
cp ../shared-source.json .build/native-shared-source.json
tar -czf /home/xander/hc-mtp/native-evidence.tar.gz \
    .build/native-*.json .build/native-*.jsonl .build/native-*.stderr \
    .build/native-clock-*.txt .build/native-*.sha256 .build/native-platform.txt \
    .build/head-build.log .build/head-tests.log \
    .build/native-first.bin .build/native-first.bin.trace
