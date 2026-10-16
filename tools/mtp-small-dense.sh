#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
tar -xf source.tar -C source
cd source
mkdir -p .build/compiler
export TMPDIR="$PWD/.build/compiler"
export XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 tools/bench-decode-fusion > .build/small-dense-build.log 2>&1 || {
    tail -80 .build/small-dense-build.log; exit 1;
}
./tools/bench-decode-fusion --dense-only > .build/small-dense-only.jsonl || {
    tail -12 .build/small-dense-only.jsonl; exit 1;
}
grep '"kind":"dense_cost"' .build/small-dense-only.jsonl
