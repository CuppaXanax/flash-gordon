#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
tar -xzf q4-tools.tar.gz
sha256sum q4_quantize.py
tail -3 source.log
