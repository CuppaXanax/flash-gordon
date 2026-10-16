#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
tar -xf fast-source.tar -C source
export TMPDIR="$PWD/.cache/work"
kind="${1:-q4}"
args=()
case "$kind" in
    q4) ;;
    q5) args=(--q5-dense) ;;
    *) echo "expected q4 or q5" >&2; exit 1 ;;
esac
.venv/bin/python source/tools/q4_quantize.py --source native-q8.gguf \
    --output "native-expert-${kind}.gguf" --ggml-library quantizer/build-native/ggml/src/libggml-base.so \
    --mtp --mtp-experts-only "${args[@]}" --threads 8 > "native-expert-${kind}-export.log" 2>&1 \
    || { tail -30 "native-expert-${kind}-export.log"; exit 1; }
tail -6 "native-expert-${kind}-export.log"
sha256sum "native-expert-${kind}.gguf"
