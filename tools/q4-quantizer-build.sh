#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
export TMPDIR="$PWD/compiler-files"
mkdir -p "$TMPDIR"
tail -8 source.log || true
ls -lh dense* || true
mkdir -p llama
tar -xf q4-llama.tar -C llama
cmake -S llama -B llama-build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release >quantizer-build.log 2>&1
cmake --build llama-build --target llama-quantize -j8 >>quantizer-build.log 2>&1 || { tail -80 quantizer-build.log; exit 1; }
test -x llama-build/bin/llama-quantize
python3 - <<'PY'
import json, urllib.request
for repo in ["unsloth/Qwen3.8-Flash-Next-GGUF", "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated"]:
    try:
        with urllib.request.urlopen("https://huggingface.co/api/models/"+repo, timeout=30) as r:
            d=json.load(r)
        print(repo, d["sha"], [x["rfilename"] for x in d["siblings"] if "imatrix" in x["rfilename"] or x["rfilename"].endswith("index.json")])
    except Exception as e: print(repo, str(e))
PY
