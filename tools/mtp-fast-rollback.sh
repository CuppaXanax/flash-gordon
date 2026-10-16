#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp/source
cp "$0" .build/fast-rollback-recipe.sh
export TMPDIR="$PWD/.build/compiler" XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 tools/bench-decode-fusion > .build/fast-rollback-build.log 2>&1 \
    || { tail -40 .build/fast-rollback-build.log; exit 1; }
./tools/bench-decode-fusion --q4 --small-n --batch 4 --small-test --rollback-test \
    --warmup 1 --iterations 1 --windows 1 > .build/fast-rollback.jsonl 2> .build/fast-rollback.stderr
tail -20 .build/fast-rollback.jsonl
sha256sum tools/bench-decode-fusion > .build/fast-rollback-binary.sha256
tar -czf /home/xander/hc-mtp/fast-evidence.tar.gz \
    .build/fast-* .build/profile-* .build/attempt-r16 .build/attempt-serial-attention
sha256sum /home/xander/hc-mtp/fast-evidence.tar.gz
