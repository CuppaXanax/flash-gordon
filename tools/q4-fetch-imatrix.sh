#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
tar -xzf q4-tools.tar.gz
python3 q4_imatrix.py >imatrix.log 2>&1 || { tail -30 imatrix.log; exit 1; }
tail -3 imatrix.log
