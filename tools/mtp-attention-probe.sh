#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
cp ../vk-fix.c src/vk.c
export TMPDIR="$PWD/.build/compiler" XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 flash-gordon tools/bench-mtp-head tests/test_fg_vk > .build/attention-build.log 2>&1 \
    || { tail -40 .build/attention-build.log; exit 1; }
./tools/bench-mtp-head --self-test
./tools/bench-mtp-head --overlay ../native-q8.gguf --embedding-f32 ../embedding.f32 \
    --replay --compare-reference --warmup 8 --iterations 40
./tools/bench-mtp-head --overlay ../native-expert-q5.gguf --embedding-f32 ../embedding.f32 \
    --head-q8 ../shared-head.q8 --replay --head-gpu --head-ways 4 \
    --warmup 512 --iterations 32 --profile
