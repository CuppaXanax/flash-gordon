#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
test -f fp16-q8.complete
sha256sum fp16-evidence-50.tar.gz fp16-logits.tar.gz fp16-q8-control.tar.gz
for service in bc250-embed bc250-rerank; do
    systemctl is-active --quiet "$service"
    port=8091
    [[ "$service" != bc250-rerank ]] || port=8092
    curl --max-time 10 -fsS "http://192.168.42.50:$port/health"
done
sha256sum -c governor-before.sha256
