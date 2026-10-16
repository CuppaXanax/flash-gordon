#!/usr/bin/env bash
set -euo pipefail
blade=$(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)
[[ $blade =~ ^5[0-3]$ ]]
cd /home/xander/flash-gordon-live/20261016-host-overhead
rm -f -- vk-wall.so cpu-wall.log host-results.tar.gz unit.log qsa.log benchmark.log
rm -f -- /home/xander/host-overhead-source.tar.gz
if [[ -d .build-scratch ]]; then rmdir .build-scratch; fi
sha256sum flash-gordon tests/test_fg_vk tests/test_qsa_prefill
echo "Offline release retained; scratch removed; no service launched."
