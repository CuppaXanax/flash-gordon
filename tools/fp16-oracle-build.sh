#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
mkdir -p hardware build
tar -xf hardware-source.tar -C hardware
export TMPDIR="$PWD/build"
make -s -C hardware -j4 tools/fp16-wire-oracle tests/test_bf16_hops > fp16-oracle-build.log 2>&1
hardware/tests/test_bf16_hops
sha256sum hardware/tools/fp16-wire-oracle hardware/tests/test_bf16_hops
