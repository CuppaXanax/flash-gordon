#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-reconcile
test -f build.ready
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
python3 source/tools/bf16_source.py ngram --manifest pack/manifest.fgm \
    --directory pack --overlay unused.gguf --placer source/tools/mtp-real-place \
    > evidence/ngram-source.log 2>&1
sha256sum pack/ngram.iq4nl > evidence/ngram.sha256
touch ngram.ready
