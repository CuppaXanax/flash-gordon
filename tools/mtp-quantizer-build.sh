#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
mkdir -p quantizer .cache/work
export TMPDIR="$PWD/.cache/work"
tar -xf ggml-source.tar -C quantizer
cmake -S quantizer -B quantizer/build-native -DBUILD_SHARED_LIBS=ON \
    -DGGML_CPU=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF > quantizer/configure.log 2>&1 \
    || { tail -60 quantizer/configure.log; exit 1; }
cmake --build quantizer/build-native --target ggml-base -j8 > quantizer/build.log 2>&1 \
    || { tail -50 quantizer/build.log; exit 1; }
find quantizer/build-native -name 'libggml-base.so*' -type f -print
