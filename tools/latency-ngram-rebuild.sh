#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat/src
tar -xf ../ngram-source.tar
mkdir -p .build
export TMPDIR=$PWD/.build XDG_CACHE_HOME=/home/xander/hc-lat/cache FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-ngram-latency tests/test_fg_vk flash-gordon > ../rebuild.log 2>&1 || { tail -50 ../rebuild.log; exit 1; }
DS4_REMOTE_TEST_FILTER=ngram_direct_lookup ./tests/test_fg_vk > ../validation-staged.txt 2>&1
cat ../validation-staged.txt
