#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
cd /home/xander/hc-experts
export XDG_CACHE_HOME="$PWD/cache"
test ! -e baseline
tar -xf source.tar
tar -xf candidate.tar tests/test_fg_vk.c
touch src/vk.c tests/test_fg_vk.c shaders/fg_moe_decode_gate_up.comp shaders/fg_moe_decode_down_reduce.comp
make -s -j8 tests/test_fg_vk tests/test_core > build.log 2>&1 || { tail -60 build.log; exit 1; }
mkdir -p baseline
cp vulkan/fg_moe_decode_gate_up.spv vulkan/fg_moe_decode_down_reduce.spv baseline/
cp shaders/fg_moe_decode_gate_up.comp shaders/fg_moe_decode_down_reduce.comp baseline/
rm -f baseline.bin
: > baseline.log
status=0
for format in q5 q4q5 q4q8; do
    FG_BENCH_EXPERT_FORMAT="$format" FG_BENCH_EXPERT_DECODE=30 FG_EXPERT_PARITY_DUMP=baseline.bin DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark ./tests/test_fg_vk >> baseline.log 2>&1 || { status=$?; break; }
done
cat baseline.log
exit "$status"
