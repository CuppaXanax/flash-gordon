#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
tar -czf physics-evidence.tar.gz physics-weights.jsonl physics-artifacts.sha256 physics-build.log
cd source
mkdir -p .build/physics-baseline
cp shaders/fg_small_dense.glsl .build/physics-baseline/
cp vulkan/fg_small_q4*.spv vulkan/fg_small_q5*.spv vulkan/fg_small_iq4*.spv .build/physics-baseline/
for n in 1 2 3 4; do
    ./tools/bench-decode-fusion --q4 --small-n --batch "$n" \
        --warmup "$((256/n))" --iterations 1 --windows 1 --profile \
        > "../physics-profile-$n.jsonl" 2> "../physics-profile-$n.stderr"
done
cd ..
tar -czf physics-evidence.tar.gz physics-weights.jsonl physics-artifacts.sha256 physics-build.log \
    physics-profile-*.jsonl physics-profile-*.stderr
sha256sum physics-evidence.tar.gz
