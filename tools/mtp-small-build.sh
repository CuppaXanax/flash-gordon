#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
mkdir -p source
tar -xf source.tar -C source
cd source
mkdir -p .build/compiler
export TMPDIR="$PWD/.build/compiler"
export XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 flash-gordon tools/bench-decode-fusion tests/test_mtp tests/test_sampler tests/test_decode_batch \
    > .build/small-build.log 2>&1 || { tail -100 .build/small-build.log; exit 1; }
./tests/test_mtp
./tests/test_sampler
./tests/test_decode_batch
bash tools/check-flags.sh
for type in 7 abc 4294967295; do
    if ./tools/bench-decode-fusion --dense-only --dense-type "$type" > .build/dense-invalid.log 2>&1; then
        echo "invalid dense type accepted: $type"; exit 1
    fi
done
for n in 1 2 3 4; do
    extra=()
    if [ "$n" = 1 ]; then extra=(--small-test --dense-bench --rollback-test); fi
    stdbuf -oL ./tools/bench-decode-fusion --q4 --small-n --batch "$n" \
        --warmup 4 --iterations 2 --windows 1 "${extra[@]}" \
        > ".build/small-smoke-${n}.jsonl" 2> ".build/small-smoke-${n}.stderr" || {
        grep '"kind":"dense_cost"' ".build/small-smoke-${n}.jsonl" || true
        tail -25 ".build/small-smoke-${n}.jsonl"
        tail -25 ".build/small-smoke-${n}.stderr"
        exit 1
    }
    grep -E '"kind":"small_block_test"|"kind":"small_causal_test"|"kind":"small_error_cleanup_test"|"kind":"rollback"' ".build/small-smoke-${n}.jsonl" || true
    tail -4 ".build/small-smoke-${n}.jsonl"
    grep '"kind":"dense_cost"' ".build/small-smoke-${n}.jsonl" || true
done
for offset in 0 1 2 3; do
    ./tools/bench-decode-fusion --q4 --small-n --batch 1 \
        --warmup "$((3+offset))" --iterations 1 --windows 1 --rollback-test \
        > ".build/small-boundary-${offset}.jsonl" 2> ".build/small-boundary-${offset}.stderr"
    grep '"kind":"rollback"' ".build/small-boundary-${offset}.jsonl"
done
sha256sum flash-gordon tools/bench-decode-fusion > .build/small-binaries.sha256
