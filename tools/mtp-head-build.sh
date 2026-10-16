#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p source
tar -xf source.tar -C source
cd source
mkdir -p .build/compiler
export TMPDIR="$PWD/.build/compiler" XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 flash-gordon tools/bench-mtp-head tests/test_mtp tests/test_sampler tests/test_decode_batch tests/test_fg_vk \
    > .build/head-build.log 2>&1 || { tail -60 .build/head-build.log; exit 1; }
./tests/test_mtp
./tests/test_sampler
./tests/test_decode_batch
bash tools/check-flags.sh
./tools/bench-mtp-head --self-test
for count in 1x -1 4294967328 ''; do
    set +e
    ./tools/bench-mtp-head --self-test --iterations "$count" > .build/head-cli-test.log 2>&1
    status=$?
    set -e
    test "$status" -eq 2
done
: > .build/head-tests.log
for filter in expert_decode_fused expert_decode_fused_q8_0 expert_decode_fused_q8_0_cooked expert_decode_fused_q8_gates; do
    DS4_REMOTE_TEST_FILTER="$filter" ./tests/test_fg_vk >> .build/head-tests.log 2>&1 \
        || { tail -40 .build/head-tests.log; exit 1; }
done
tail -16 .build/head-tests.log
