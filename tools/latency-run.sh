#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
mkdir -p build cache
export TMPDIR=$PWD/build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar
gcc -O2 -std=c17 -Wall -Wextra -Werror tools/latency-vk.c -lvulkan -o latency-vk
gcc -O2 -std=c17 -Wall -Wextra -Werror tools/latency-storage.c -pthread -o latency-storage
for spec in empty:0 dependent:1 reduction:2 read:3 chase:4 subgroup:5; do
    name=${spec%:*}; mode=${spec#*:}
    glslangValidator -V --target-env vulkan1.1 -DMODE="$mode" tools/latency.comp -o "$name.spv" > compile.log 2>&1 || { cat compile.log; exit 1; }
done
{
    for n in 0 1 8 64 256; do
        ./latency-vk empty.spv 40960 1 "$n" 0 0
        ./latency-vk empty.spv 40960 1 "$n" 1 0
        ./latency-vk empty.spv 40960 0 "$n" 1 0
        ./latency-vk empty.spv 40960 0 "$n" 0 1
        ./latency-vk dependent.spv 40960 1 "$n" 1 0
    done
    for groups in 1 2 4 8 16 24 32 64 128 256 1024; do
        ./latency-vk empty.spv 40960 "$groups" 1 0 0
        ./latency-vk reduction.spv 40960 "$groups" 1 1 0
        ./latency-vk subgroup.spv 40960 "$groups" 1 1 0
    done
    for bytes in 4096 40960 358400 1048576 8388608; do
        ./latency-vk read.spv "$bytes" 1 1 1 0
        ./latency-vk read.spv "$bytes" 256 1 1 0
        ./latency-vk chase.spv "$bytes" 1 1 1 0
    done
} > gpu.jsonl
cat gpu.jsonl
