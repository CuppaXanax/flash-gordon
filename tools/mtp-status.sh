#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
cat .build/smoke-n1.jsonl .build/smoke-n2.jsonl | grep -E '"kind":"(rollback|parity)"'
test -f .build/measure/summary.json
