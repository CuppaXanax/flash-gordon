#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
mkdir -p pack
cp manifest.fgm pack/manifest.fgm
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
python3 source/tools/bf16_source.py metadata --manifest pack/manifest.fgm \
    --directory pack --overlay overlay.gguf
python3 source/tools/bf16_source.py ngram --manifest pack/manifest.fgm \
    --directory pack --overlay overlay.gguf --placer source/tools/mtp-real-place \
    > ngram-source.log 2>&1
python3 source/tools/bf16_source.py 0 --manifest pack/manifest.fgm \
    --directory pack --overlay overlay.gguf --placer source/tools/mtp-real-place \
    > rank0-source.log 2>&1
