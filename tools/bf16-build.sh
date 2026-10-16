#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
mkdir -p source build cache
tar -xf source.tar -C source
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
make -s -C source -j8 tools/bench-bf16-hops tools/mtp-real-place tests/test_bf16_hops
source/tests/test_bf16_hops
sha256sum source/tools/bench-bf16-hops source/tests/test_bf16_hops
