#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
tail -12 source.log 2>/dev/null || true
tail -5 quantizer-build.log 2>/dev/null || true
ls -lh dense* 2>/dev/null || true
for log in source-parity.log quantize-q4.log quantize-protected.log; do
    if [ -f "$log" ]; then echo "=== $log ==="; tail -4 "$log"; fi
done
