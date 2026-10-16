#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
sudo -n prlimit --pid "$$" --memlock=268435456:268435456
mkdir -p source .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
tar -xf source.tar -C source
cd source
mkdir -p .build
make -j8 all test-mtp test-multi-session tools/bench-mtp-head \
    tools/bench-decode-fusion > ../serving-build.log 2>&1 \
    || { tail -80 ../serving-build.log; exit 1; }
bash tools/check-flags.sh
./tests/test_mtp_embedding .build/mtp-embedding-gpu-test --gpu \
    > ../serving-embedding-gpu.log 2>&1
tail -1 ../serving-embedding-gpu.log
python3 -m unittest discover -s tests -p 'test_mtp_*.py'
bash -n tools/mtp-lease-guard.sh
./tools/bench-mtp-head --self-test > ../serving-head-selftest.jsonl
./tools/bench-decode-fusion --q4 --small-n --batch 2 --session-test \
    --small-test --rollback-test --warmup 1 --iterations 1 --windows 1 \
    > ../serving-small-sessions.jsonl 2> ../serving-small-sessions.stderr
if [[ -f ../native-q8.gguf ]]; then
    ./tools/bench-mtp-head --overlay ../native-q8.gguf --session-test \
        > ../serving-native-sessions.jsonl 2> ../serving-native-sessions.stderr
fi
cat ../serving-head-selftest.jsonl
grep -E 'small_n_sessions|small_block_test|small_causal_test|rollback' \
    ../serving-small-sessions.jsonl
if [[ -f ../serving-native-sessions.jsonl ]]; then
    cat ../serving-native-sessions.jsonl
fi
