#!/usr/bin/env bash
set -euo pipefail
bash /home/xander/hc-mtp/source/tools/mtp-fast-build.sh
cd /home/xander/hc-mtp
for attempt in $(seq 1 60); do
    if [ -s native-source.json ] && [ -s shared-source.json ]; then break; fi
    sleep 5
done
test -s native-source.json
test -s shared-source.json
bash source/tools/mtp-fast-probe.sh
