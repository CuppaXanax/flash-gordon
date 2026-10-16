#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname -I)" == *192.168.42.53* ]]
source=/home/xander/hc-q4/production-dense-q8.gguf
[[ $(stat -c %s "$source") == 3801939104 ]]
printf '%s  %s\n' 918a6175bf6082863b2306c125802970164cbe9777847997e5c1c51aa3096684 "$source" | sha256sum -c -
printf 'READ_ONLY_Q8_EXPORT %s\n' "$(date -u +%FT%TZ)"
