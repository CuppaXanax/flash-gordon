#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p source .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
tar -xf source.tar -C source
cd source
mkdir -p .build
make -j8 all tools/mtp-acceptance tests/test_core test-mtp-capture test-mtp test-sampler test-decode-batch > ../capture-build.log 2>&1 \
    || { tail -60 ../capture-build.log; exit 1; }
./tests/test_core > ../capture-core-tests.log
bash tools/check-flags.sh
./flash-gordon --help | head -14
