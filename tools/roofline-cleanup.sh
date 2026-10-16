#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank
cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
/usr/local/bin/bc250-cu-live-manager status | grep 'CUs active & routed  : 40/40'
curl --max-time 10 -fsS http://192.168.42.50:8091/health
curl --max-time 10 -fsS http://192.168.42.50:8092/health
cd /home/xander
rm -rf /home/xander/hc-roof
test ! -e /home/xander/hc-roof
echo 'hc-roof absent; services healthy; original governor and CU profile restored'
