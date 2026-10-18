#!/usr/bin/env bash
set -euo pipefail
blade=$(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)
[[ $blade =~ ^5[0-3]$ ]]
release=/home/xander/flash-gordon-live/20261016-host-overhead
mkdir -p "$release"
cd "$release"
tar -xzf /home/xander/host-overhead-source.tar.gz
mkdir -p .build-scratch
export TMPDIR="$PWD/.build-scratch"
chmod +x glslc
make -j6 all tests/test_fg_vk tests/test_qsa_prefill >build.log 2>&1 || { tail -80 build.log; exit 1; }
bash tools/check-flags.sh
sha256sum flash-gordon
rmdir .build-scratch
