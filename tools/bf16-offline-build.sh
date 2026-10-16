#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
if [[ -f prepared.sha256 && ! -f prepared-binary.sha256 ]]; then
    sha256sum source/tools/bf16-offline > prepared-binary.sha256
fi
tar -xf offline-source.tar -C source
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
make -s -C source -j8 tools/bf16-offline > offline-build.log 2>&1
binary=$(sha256sum source/tools/bf16-offline)
for proof in stage-*.sha256; do
    [[ -f "$proof" ]] || continue
    if [[ "$(head -n 1 "$proof")" != "$binary" ]]; then
        mv "$proof" "$proof.superseded"
    fi
done
source/tools/bf16-offline plan manifest.fgm
sha256sum source/tools/bf16-offline
