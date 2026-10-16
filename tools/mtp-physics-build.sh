#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp
mkdir -p source .cache/compiler
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
tar -xf source.tar -C source
cd source
mkdir -p .build
make -j8 tools/bench-decode-fusion > ../physics-build.log 2>&1 \
    || { tail -70 ../physics-build.log; exit 1; }
./tools/bench-decode-fusion --q4 --ledger-only > ../physics-weights.jsonl
sha256sum tools/bench-decode-fusion ../source.tar > ../physics-artifacts.sha256
cat ../physics-weights.jsonl
for pack in /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04 \
    /home/xander/fg-ring-pack-abliterated; do
    if [[ -d "$pack" ]]; then
        printf 'AVAILABLE_PACK %s\n' "$pack"
        find "$pack" -maxdepth 1 -type f -printf '%f %s\n'
    fi
done
