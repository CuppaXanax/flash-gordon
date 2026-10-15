set -euo pipefail
cd /home/xander/hc-q4
mkdir -p compact
tar -xf compact.tar -C compact
cd compact
export MESA_SHADER_CACHE_DIR="$PWD/.mesa-cache"
make -j8 tests/test_fg_vk > build.log 2>&1 || { tail -70 build.log; exit 1; }
DS4_REMOTE_TEST_FILTER=dense_quant_compact ./tests/test_fg_vk
export FG_BENCH_DENSE_COMPACT=1 FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_SKIP_HEAD=1 FG_BENCH_DENSE_Q4_ONLY=1
for n in 1 8 32 128 512; do
  printf '\nTOKENS=%s\n' "$n"
  FG_BENCH_DENSE_N=$n DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee "bench-n${n}.log"
done
for v in r4w32 r8w32 r4w128 r8w128; do
  printf '\nIQ_VARIANT=%s\n' "$v"
  FG_BENCH_IQ_ONLY=1 FG_BENCH_IQ_VARIANT=$v FG_BENCH_DENSE_N=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1
done
