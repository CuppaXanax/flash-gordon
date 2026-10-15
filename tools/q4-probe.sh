#!/bin/bash
set -euo pipefail
mkdir -p /home/xander/hc-q4
cd /home/xander/hc-q4
hostname
df -h .
command -v cc cmake glslc python3
ls -d /home/xander/*llama* /home/xander/*flash* 2>/dev/null || true
python3 - <<'PY'
import json, urllib.request
repo="huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF"
try:
    with urllib.request.urlopen("https://huggingface.co/api/models/"+repo) as r:
        data=json.load(r)
    print(json.dumps({"sha": data["sha"], "files": [f["rfilename"] for f in data["siblings"]]}, indent=2))
except Exception as e:
    print(type(e).__name__, str(e))
PY
