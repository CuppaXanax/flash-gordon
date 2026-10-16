#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
for file in native-q8.gguf shared-head.q8 native-source.json shared-source.json; do
    if [ -f "$file" ]; then ls -l "$file"; fi
done
tail -3 native-export.log shared-export.log native-setup.log shared-setup.log 2>/dev/null || true
ps -eo pid,etimes,args | grep -E '[m]tp_source.py|[m]tp_shared_source.py' || true
