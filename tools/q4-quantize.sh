#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
tar -xzf q4-tools.tar.gz
python3 - <<'PY'
import json, urllib.request
repo="unsloth/Qwen3.8-Flash-Next-GGUF"
try:
    with urllib.request.urlopen("https://huggingface.co/api/models/"+repo, timeout=30) as r:
        d=json.load(r)
    print(repo, d["sha"], [x["rfilename"] for x in d["siblings"] if "imatrix" in x["rfilename"]])
except Exception as e: print(repo, str(e))
PY
while [ ! -f dense-bf16.gguf ]; do
    if ! pgrep -f '^python3 q4_source.py' >/dev/null; then tail -30 source.log; exit 1; fi
    sleep 30
done
python3 q4_check_source.py --bf16 dense-bf16.gguf --output source-parity.json >source-parity.log 2>&1 || { tail -40 source-parity.log; exit 1; }
python3 q4_imatrix.py >imatrix.log 2>&1 || { tail -30 imatrix.log; exit 1; }
python3 q4_quantize.py --source dense-bf16.gguf --output dense-q4.gguf --ggml-library llama-build/bin/libggml-base.so --imatrix dense-imatrix.gguf >quantize-q4.log 2>&1
python3 q4_quantize.py --source dense-bf16.gguf --output dense-protected.gguf --ggml-library llama-build/bin/libggml-base.so --protected --imatrix dense-imatrix.gguf >quantize-protected.log 2>&1
sha256sum dense-*.gguf | tee dense-sha256.txt
ls -lh dense-*.gguf
