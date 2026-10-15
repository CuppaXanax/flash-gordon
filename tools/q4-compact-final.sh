set -euo pipefail
cd /home/xander/hc-q4
mkdir -p compact
tar -xf compact.tar -C compact
cd compact
export MESA_SHADER_CACHE_DIR="$PWD/.mesa-cache"
make -j8 tests/test_fg_vk > build.log 2>&1 || { tail -60 build.log; exit 1; }
DS4_REMOTE_TEST_FILTER=dense_quant_compact ./tests/test_fg_vk
export FG_BENCH_DENSE_COMPACT=1 FG_BENCH_DENSE_SELECTED=1 FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_SKIP_HEAD=1 FG_BENCH_DENSE_Q4_ONLY=1
for n in 1 8 16 32 64 128 256 512; do
  printf '\nTOKENS=%s\n' "$n"
  FG_BENCH_DENSE_N=$n DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1 | tee "final-n${n}.log"
done
for repeat in 2 3; do
  printf '\nGEMV_REPEAT=%s\n' "$repeat"
  FG_BENCH_DENSE_N=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1
done
printf '\nPROTECTED_Q5=1\n'
unset FG_BENCH_DENSE_Q4_ONLY
FG_BENCH_DENSE_N=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk 2>&1
