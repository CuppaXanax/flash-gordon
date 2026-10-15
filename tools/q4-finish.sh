#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4
python3 - <<'PY'
import json
from q4_source import REPO, REV
d=json.load(open("source-parity.json"))
assert d["passed"] and d["repo"] == REPO and d["revision"] == REV
PY
imatrix=()
if [ -f dense-imatrix.gguf ]; then imatrix=(--imatrix dense-imatrix.gguf); fi
python3 q4_quantize.py --source dense-bf16.gguf --output dense-q4.gguf "${imatrix[@]}" \
    --ggml-library llama-build/bin/libggml-base.so >quantize-q4.log 2>&1
python3 q4_quantize.py --source dense-bf16.gguf --output dense-protected.gguf "${imatrix[@]}" \
    --ggml-library llama-build/bin/libggml-base.so --protected >quantize-protected.log 2>&1
sha256sum dense-*.gguf | tee dense-sha256.txt
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
hashes={line.split()[1]:line.split()[0] for line in Path("dense-sha256.txt").read_text().splitlines()}
artifacts=[]
for name in ("dense-bf16", "dense-q4", "dense-protected"):
    path=Path(name+".gguf")
    d=json.loads(path.with_suffix(".json").read_text())
    rows=d["tensors"]
    types=Counter()
    for t in rows:
        types[str(t["type"])]+=t["bytes"]
    artifacts.append(dict(path=str(path.resolve()), bytes=path.stat().st_size,
        sha256=hashes[path.name], tensors=len(rows), tensor_bytes=sum(types.values()),
        bytes_by_type=dict(types),
        router_types=dict(Counter(str(t["type"]) for t in rows
            if t["name"].endswith(".ffn_gate_inp.weight"))),
        imatrix=d.get("imatrix"),
        lossless_router_conversions=sum(bool(t.get("lossless_bf16")) for t in rows)))
Path("artifacts.json").write_text(json.dumps(artifacts,indent=2)+"\n")
print(json.dumps(artifacts,indent=2))
PY
