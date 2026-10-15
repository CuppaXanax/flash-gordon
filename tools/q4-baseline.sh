#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-q4
export TMPDIR="$PWD/compiler-files"
mkdir -p "$TMPDIR"
mkdir -p baseline
tar -xf q4-baseline.tar -C baseline
cd code
cc -O3 -g -std=c17 -D_GNU_SOURCE -Iinclude -Ivendor \
    ../baseline/src/vk.c ../baseline/src/output.c ../baseline/src/model.c \
    ../baseline/tests/test_fg_vk.c \
    src/quant.o src/q38_math.o src/ngram.o src/uring.o src/sha256.o \
    src/q38_schema.o src/gguf.o src/topology.o src/util.o src/loader.o \
    src/manifest.o src/runtime_options.o src/sampler.o src/protocol.o \
    -pthread -lm -lvulkan -o ../baseline/test_fg_vk
set +e
DS4_REMOTE_TEST_FILTER=qsa_record_commit ../baseline/test_fg_vk >../baseline-qsa.log 2>&1
rc=$?
cat ../baseline-qsa.log
echo "ORIGINAL_BASELINE_QSA_EXIT=$rc"
exit 0
