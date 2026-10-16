#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
tar -xf fast-source.tar -C source
export TMPDIR="$PWD/.cache/work"
.venv/bin/python source/tests/test_mtp_quantize.py
.venv/bin/python source/tools/q4_quantize.py --source native-q8.gguf \
    --output native-q4.gguf --ggml-library quantizer/build-native/ggml/src/libggml-base.so \
    --mtp --threads 8 > native-q4-export.log 2>&1 || { tail -30 native-q4-export.log; exit 1; }
tail -6 native-q4-export.log
sha256sum native-q4.gguf
