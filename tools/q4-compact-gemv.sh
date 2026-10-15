set -euo pipefail
cd /home/xander/hc-q4
tar -xf compact.tar -C compact
cd compact
export MESA_SHADER_CACHE_DIR="$PWD/.mesa-cache"
make -j8 tests/test_fg_vk > build.log 2>&1 || { tail -60 build.log; exit 1; }
DS4_REMOTE_TEST_FILTER=dense_quant_compact ./tests/test_fg_vk
FG_BENCH_DENSE_COMPACT=1 FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_SKIP_HEAD=1 FG_BENCH_DENSE_Q4_ONLY=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1
