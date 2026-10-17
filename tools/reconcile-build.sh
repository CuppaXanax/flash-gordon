#!/usr/bin/env bash
set -euo pipefail
root=/home/xander/hc-reconcile
cd "$root"
tar -xf source.tar -C source
tar -xf patch.tar -C source
tar -xzf metadata.tar.gz
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
if [[ -f prepared.ready ]]; then
    cp evidence/binaries.sha256 evidence/prepare-binaries.sha256
fi
touch source/tools/bf16_offline.c
make -s -C source -j12 tools/bf16-offline tools/mtp-real-place > evidence/build.log 2>&1
sha256sum source/tools/bf16-offline source/tools/mtp-real-place > evidence/binaries.sha256
cp pack/manifest.fgm evidence/mixed-manifest.fgm
cp original-q8.fgm evidence/original-q8.fgm
sha256sum /etc/cyan-skillfish-governor-smu/config.toml > evidence/governor-before.sha256
cat /sys/class/drm/card*/device/power_dpm_force_performance_level > evidence/performance-before.txt
touch build.ready
