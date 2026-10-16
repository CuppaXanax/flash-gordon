#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
for i in $(seq 1 120); do
    [[ ! -f fp16-evidence.complete ]] || break
    [[ "$i" != 120 ]] || { echo 'quality evidence not ready' >&2; exit 1; }
    sleep 5
done
sha256sum fp16-evidence-50.tar.gz fp16-logits.tar.gz
