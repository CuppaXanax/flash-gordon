#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
python3 q4_source.py --output dense-bf16.gguf 2>&1 | tee source.log
