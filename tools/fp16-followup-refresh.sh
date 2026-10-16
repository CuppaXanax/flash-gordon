#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
test ! -f fp16.complete
tar -xf fp16-followup.tar
sha256sum helpers/*.sh
