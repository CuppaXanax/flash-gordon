#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-abq-stack-live-20261016
mkdir -p cache compiler
export TMPDIR="$PWD/compiler" MESA_SHADER_CACHE_DIR="$PWD/cache" XDG_CACHE_HOME="$PWD/cache"
export FG_SHADER_DIR=/home/xander/flash-gordon-live/20261016-abq-stack-adaptive/vulkan
for test in roof_controls roof_dense_exact roof_qsa roof_hc_up_mix roof_hc_down_inject roof_gdn_gate_controls expert_decode_fused_q8_gates submit_group; do
    DS4_REMOTE_TEST_FILTER="$test" /home/xander/flash-gordon-live/20261016-abq-stack-adaptive/tests/test_fg_vk
done
