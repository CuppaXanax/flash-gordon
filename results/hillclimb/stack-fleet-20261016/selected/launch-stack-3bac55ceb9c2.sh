#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\n' 'f35d2b228e81e4d8a2b601085055703978c830d39f975fa27278cbf8667ea42d' '/home/xander/flash-gordon-live/20261015-abq-operator/ab-instance-stack-3bac55ceb9c2.sh' | sha256sum -c -
bash '/home/xander/flash-gordon-live/20261015-abq-operator/ab-instance-stack-3bac55ceb9c2.sh' "${1:?start,stop,status}" '/home/xander/flash-gordon-live/20261016-abq-stack-adaptive' '/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm' '2' '1'