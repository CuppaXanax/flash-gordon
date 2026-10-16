#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
mkdir -p source .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
tar -xf source.tar -C source
cd source
mkdir -p .build
make -j8 tools/bench-decode-fusion > ../verify-build.log 2>&1 \
    || { tail -70 ../verify-build.log; exit 1; }
make test-mtp-capture > ../verify-capture.log 2>&1 \
    || { cat ../verify-capture.log; exit 1; }
python3 -m unittest discover -s tests -p test_mtp_physics.py
./tools/bench-decode-fusion --q4 --small-n --batch 2 --small-test --rollback-test \
    --warmup 1 --iterations 1 --windows 1 > ../verify-gates.jsonl 2> ../verify-gates.stderr \
    || { tail -15 ../verify-gates.jsonl; cat ../verify-gates.stderr; exit 1; }
grep -E 'small_block_test|small_causal_test|rollback' ../verify-gates.jsonl
./tools/bench-decode-fusion --q4 --small-n --batch 1 \
    --warmup 256 --iterations 64 --windows 3 --routing-trace ../physics-synthetic-routes.jsonl \
    > ../physics-routing-run.jsonl 2> ../physics-routing-run.stderr
bash tools/mtp-serving-bench.sh
