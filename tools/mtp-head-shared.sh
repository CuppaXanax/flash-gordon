#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
export OPENBLAS_NUM_THREADS=1
.venv/bin/python source/tools/mtp_shared_source.py --head shared-head.q8 \
    --embedding embedding.f32 --audit shared-source.json > shared-export.log 2>&1
sha256sum shared-head.q8 embedding.f32
tail -n 2 shared-export.log
