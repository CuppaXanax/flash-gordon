#!/bin/bash
set -euo pipefail
cd /home/xander/hc-fusion/src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
unset FG_DECODE_FUSION
for filter in dense_quant_soa dense_quant_compact q8_decode_shape_parity \
    batch_submission_parity pipeline_flush_parity static_batch_replay; do
    DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk
done > ../fixed/validation-default.txt 2>&1
systemctl is-active --quiet cyan-skillfish-governor-smu
cmp ../fixed/original.toml /etc/cyan-skillfish-governor-smu/config.toml
cat ../fixed/validation-default.txt
