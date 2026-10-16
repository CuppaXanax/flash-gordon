#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
mkdir -p src build cache
export TMPDIR=$PWD/build XDG_CACHE_HOME=$PWD/cache
tar -xf full-source.tar -C src
cd src
mkdir -p .build
export FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tests/test_fg_vk tools/bench-ngram-latency tools/bench-fabric-latency > ../build.log 2>&1 || { tail -60 ../build.log; exit 1; }
bash tools/check-flags.sh
DS4_REMOTE_TEST_FILTER=ngram_direct_lookup ./tests/test_fg_vk > ../validation.txt 2>&1
cat ../validation.txt
cp tools/bench-fabric-latency ../bench-fabric-latency
