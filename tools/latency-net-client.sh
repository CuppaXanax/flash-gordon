#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
python3 tools/latency-net.py client > network.jsonl
cat network.jsonl
