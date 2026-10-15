#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
files=()
for file in kernel-rates.log kernel-rates-cooked.log build.log core.log \
    regression.log standalone-core.log standalone-vulkan.log baseline-qsa.log \
    dense-bf16.json dense-q4.json dense-protected.json dense-imatrix.json artifacts.json source-parity.json \
    source-parity.log dense-sha256.txt imatrix.log; do
    if [ -f "$file" ]; then files+=("$file"); fi
done
tar -czf evidence.tar.gz "${files[@]}"
ls -lh evidence.tar.gz
