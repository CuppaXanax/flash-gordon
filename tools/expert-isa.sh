#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
cd /home/xander/hc-experts
export XDG_CACHE_HOME="$PWD/cache"
RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true DS4_REMOTE_TEST_FILTER=expert_decode_fused_wide ./tests/test_fg_vk > isa.log 2>&1
gzip -c isa.log > isa.log.gz
cp vulkan/fg_moe_decode_down_reduce.spv isa-candidate-down.spv
trap 'cp isa-candidate-down.spv vulkan/fg_moe_decode_down_reduce.spv' EXIT
cp baseline/fg_moe_decode_down_reduce.spv vulkan/
RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true DS4_REMOTE_TEST_FILTER=expert_decode_fused_wide ./tests/test_fg_vk > isa-baseline.log 2>&1
gzip -c isa-baseline.log > isa-baseline.log.gz
printf 'Candidate and original Q5 ISA captured; subgroup sizes:\n'
grep 'api_subgroup_size:' isa.log | sort -u
