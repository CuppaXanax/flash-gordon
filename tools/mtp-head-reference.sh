#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
export OPENBLAS_NUM_THREADS=4
.venv/bin/python source/tools/mtp_head_reference.py \
    --overlay native-q8.gguf --head-q8 shared-head.q8 \
    --dump source/.build/native-first.bin --output source/.build/native-reference.json
