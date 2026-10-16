#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p .cache/work .cache/pip
export TMPDIR="$PWD/.cache/work" PIP_CACHE_DIR="$PWD/.cache/pip"
export OPENBLAS_NUM_THREADS=1
if ! .venv/bin/python -c 'import numpy' 2>/dev/null; then
    python3 -m venv .venv
    .venv/bin/python -m pip install --disable-pip-version-check \
        --index-url https://packagefeedproxy.microsoft.io/pypi/simple numpy
fi
.venv/bin/python mtp_source.py --revision 298f94632b784e26a7fe576114f82066689d5baa \
    --audit native-source.json --pack native-q8.gguf > native-export.log 2>&1
sha256sum native-q8.gguf
tail -n 3 native-export.log
