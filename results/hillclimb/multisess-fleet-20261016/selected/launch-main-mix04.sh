#!/usr/bin/env bash
set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
if [[ ${1:-start} == start ]]; then
  printf '%s  %s\n' '1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789' '/home/xander/flash-gordon-live/20261015-abq-main/flash-gordon' | sha256sum -c -
  printf '%s  %s\n' '0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252' '/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm' | sha256sum -c -
fi
exec bash "$run/ab-instance-multisess.sh" "${1:-start}" '/home/xander/flash-gordon-live/20261015-abq-main' '/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm' '1' '0'
