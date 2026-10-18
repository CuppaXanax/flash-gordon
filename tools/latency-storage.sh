#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
./latency-storage /home/xander/hc-lat/synthetic.iq4nl > storage.jsonl
cat storage.jsonl
