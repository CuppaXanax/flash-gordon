#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
cc -O2 mtp_dot.c -lvulkan -o mtp-dot
./mtp-dot
