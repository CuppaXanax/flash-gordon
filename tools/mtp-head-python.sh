#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p .cache/work
export TMPDIR="$PWD/.cache/work"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check \
    --cache-dir .cache/pip --index-url https://packagefeedproxy.microsoft.io/pypi/simple numpy
