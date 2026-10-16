#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
mkdir -p source build cache
tar -xf source.tar -C source
python3 - <<'PY'
from pathlib import Path
for p in Path("source/tools").glob("*.sh"):
    p.write_bytes(p.read_bytes().replace(b"\r\n",b"\n"))
PY
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
make -s -C source -j8 tools/bench-bf16-hops tools/bf16-offline tests/test_bf16_hops \
    > fp16-build.log 2>&1
source/tests/test_bf16_hops
sha256sum source/tools/bench-bf16-hops source/tools/bf16-offline source/tests/test_bf16_hops
