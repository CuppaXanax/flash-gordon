#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
sha256sum flash-gordon tools/bench-decode-fusion > .build/mtp-binary-sha256.txt
tar -cf /home/xander/hc-mtp/evidence.tar .build/measure/*.jsonl \
    .build/measure/*.txt .build/measure/*.stderr .build/measure/summary.json \
    .build/rollback-*.jsonl .build/rollback-*.stderr .build/validation.json \
    .build/mtp-binary-sha256.txt .build/validate-build.log
