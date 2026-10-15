set -euo pipefail
cd /home/xander/hc-q4
mkdir -p kernels
tar -xf kernels.tar -C kernels
cd kernels
export MESA_SHADER_CACHE_DIR="$PWD/.mesa-cache"
touch shaders/fg_dense_soa_impl.glsl
make -j8 tests/test_fg_vk tests/test_core > build.log 2>&1 || { tail -70 build.log; exit 1; }
export FG_BENCH_DENSE_SOA=1
DS4_REMOTE_TEST_FILTER=dense_quant ./tests/test_fg_vk 2>&1
export FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_SKIP_HEAD=1
export FG_BENCH_DENSE_Q4_ONLY=1
for n in 1 8 16 32 64 128 256 512; do
  printf '\nTOKENS=%s\n' "$n"
  FG_BENCH_DENSE_N=$n DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee "bench-n${n}.log"
done
for repeat in 2 3; do
  printf '\nGEMV_REPEAT=%s\n' "$repeat"
  FG_BENCH_DENSE_N=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee "gemv-repeat${repeat}.log"
done
unset FG_BENCH_DENSE_SKIP_HEAD
FG_BENCH_DENSE_ONLY_HEAD=1 FG_BENCH_DENSE_N=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee head.log
