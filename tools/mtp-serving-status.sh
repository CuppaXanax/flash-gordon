#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
for file in serving-head-selftest.jsonl serving-small-sessions.jsonl \
    serving-small-sessions.stderr serving-native-sessions.jsonl \
    serving-native-sessions.stderr; do
    if [[ -f "$file" ]]; then
        echo "== $file"
        tail -15 "$file"
    fi
done
