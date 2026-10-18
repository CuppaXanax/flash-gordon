#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat/src
mkdir -p .build
export TMPDIR=$PWD/.build XDG_CACHE_HOME=/home/xander/hc-lat/cache FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tests/test_fg_vk flash-gordon > ../final-build.log 2>&1 || { tail -50 ../final-build.log; exit 1; }
{
    DS4_REMOTE_TEST_FILTER=ngram_direct_lookup ./tests/test_fg_vk
    DS4_REMOTE_TEST_FILTER=iq4_nl_dequant ./tests/test_fg_vk
} > ../validation-final.txt 2>&1
cat ../validation-final.txt
if find . -maxdepth 1 -name '.fg-ngram-*' | grep -q .; then exit 1; fi
