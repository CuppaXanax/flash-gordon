#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p source
tar -xf source.tar -C source
bash source/tools/mtp-head-python.sh
cp source/tools/mtp_source.py mtp_source.py
bash source/tools/mtp-head-export.sh > native-setup.log 2>&1 &
native=$!
bash source/tools/mtp-head-shared.sh > shared-setup.log 2>&1 &
shared=$!
status=0
wait "$native" || status=1
wait "$shared" || status=1
cat native-setup.log shared-setup.log
exit "$status"
