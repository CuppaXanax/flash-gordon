#!/bin/bash
set -eu
cd /home/xander/hc-mtpk
hostname
systemctl show -p ActiveState -p SubState -p Result -p MainPID bc250-embed bc250-rerank || true
systemctl is-active cyan-skillfish-governor-smu
sha256sum evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml
ps -eo pid,args | grep '[c]yan-skillfish-governor-smu' || true
ss -lntp | grep -E ':80[0-9][0-9] |:30[0-9][0-9] ' || true
for f in sweep/oracles.stderr sweep/build.txt; do
    if [ -f "$f" ]; then echo "$f"; tail -8 "$f"; fi
done
if [ -f sweep/oracles.jsonl ]; then tail -3 sweep/oracles.jsonl; fi
items=(baseline evidence)
if [ -d sweep ]; then items+=(sweep); fi
tar -czf diagnostics.tar.gz --exclude=baseline/vulkan --exclude=baseline/bench \
    --exclude=sweep/governor.log "${items[@]}"
