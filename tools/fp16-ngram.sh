#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
tar -xzf original-evidence.tar.gz prompts.fgev \
    pack/source-0.jsonl pack/source-1.jsonl pack/source-2.jsonl pack/source-3.jsonl \
    pack/source-4.jsonl pack/source-5.jsonl pack/source-6.jsonl pack/source-7.jsonl
python3 source/tools/bf16_source.py ngram --manifest pack/manifest.fgm \
    --directory pack --overlay unused.gguf --placer source/tools/mtp-real-place \
    > fp16-ngram-source.log 2>&1
sha256sum pack/ngram.iq4nl
touch fp16-ngram.ready
