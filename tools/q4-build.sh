#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-q4
export TMPDIR="$PWD/compiler-files"
mkdir -p "$TMPDIR"
mkdir -p code
tar -xzf q4-code.tar.gz -C code
find code/tools -name '*.sh' -exec sed -i 's/\r$//' {} +
cd code
make -j8 flash-gordon tools/q4-repack tests/test_fg_vk tests/test_core shaders >../build.log 2>&1 || { tail -100 ../build.log; exit 1; }
bash tools/q4-safe-tests.sh make tests/test_core >../core.log 2>&1
./tests/test_core >>../core.log 2>&1 || { tail -60 ../core.log; exit 1; }
tail -1 ../core.log
DS4_REMOTE_TEST_FILTER=dense_quant ./tests/test_fg_vk
DS4_REMOTE_TEST_FILTER=dense_quant_cooked ./tests/test_fg_vk
DS4_REMOTE_TEST_FILTER=router_bf16_exact ./tests/test_fg_vk
FG_BENCH_DENSE_QUANT=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee ../kernel-rates.log
FG_BENCH_DENSE_COOKED=1 FG_BENCH_DENSE_QUANT=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee ../kernel-rates-cooked.log
