#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
tar -xf source.tar
trap 'firewall-cmd --remove-port=19552/tcp >/dev/null 2>&1 || true' EXIT
firewall-cmd --add-port=19552/tcp >/dev/null
python3 tools/latency-net.py server
