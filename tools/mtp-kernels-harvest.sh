#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
find round2 round3 -maxdepth 1 -type f -printf '%p %s\n' 2>/dev/null || true
for f in round2/build.txt round2/cpu-f2048.txt round2/oracles-f2314.stderr round3/build.txt round3/qsa-cpu.txt; do
    if [ -f "$f" ]; then echo "=== $f"; tail -12 "$f"; fi
done
tar -czf harvest.tar.gz evidence round*
