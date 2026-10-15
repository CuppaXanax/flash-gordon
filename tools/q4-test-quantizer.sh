#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
tar -xzf q4-tools.tar.gz
FG_Q4_GGML_LIBRARY="$PWD/llama-build/bin/libggml-base.so" \
    python3 -m unittest test_q4_tools.py
