#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
find . -maxdepth 1 -type f -printf '%f %s\n'
for log in native-export.log shared-export.log; do
    if [ -f "$log" ]; then echo "$log"; tail -n 3 "$log"; fi
done
if [ -d source/.build ]; then
    for log in source/.build/native-[123].stderr; do
        if [ -f "$log" ]; then echo "$log"; tail -n 12 "$log"; fi
    done
fi
