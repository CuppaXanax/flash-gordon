#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
df -h .
free -h
find /home/xander -maxdepth 3 -type f \( -name manifest.fgm -o -name token-embedding.q8_0 -o -name 'mtp*.gguf' \) -printf '%p %s\n'
python3 -c 'import numpy; print("numpy", numpy.__version__)'
