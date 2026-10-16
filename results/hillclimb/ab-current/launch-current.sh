#!/usr/bin/env bash
set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
if [[ ${1:-start} == start ]]; then
  printf '%s  %s\n' 'e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2' '/home/xander/flash-gordon-live/20260926-abl2/flash-gordon' | sha256sum -c -
  printf '%s  %s\n' '35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb' '/home/xander/fg-ring-pack-abliterated/manifest.fgm' | sha256sum -c -
fi
exec bash "$run/ab-instance.sh" "${1:-start}" '/home/xander/flash-gordon-live/20260926-abl2' '/home/xander/fg-ring-pack-abliterated/manifest.fgm' '1' '0'
