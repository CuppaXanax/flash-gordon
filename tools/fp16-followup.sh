#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
tar -xf fp16-followup.tar
for i in $(seq 1 180); do
    if [[ -f fp16.complete ]] &&
       systemctl is-active --quiet bc250-embed &&
       systemctl is-active --quiet bc250-rerank &&
       curl --max-time 3 -fsS http://192.168.42.50:8091/health >/dev/null 2>&1 &&
       curl --max-time 3 -fsS http://192.168.42.50:8092/health >/dev/null 2>&1; then
        break
    fi
    [[ "$i" != 180 ]] || { echo 'primary numerical run did not finish' >&2; exit 1; }
    sleep 60
done
sleep 10
test -f fp16-q8-source.ready
bash helpers/fp16-offline-run.sh > fp16-resume.log 2>&1
bash helpers/fp16-real-wire-oracle.sh > fp16-real-wire-oracle.log 2>&1
bash helpers/fp16-evidence.sh > fp16-evidence.log 2>&1
bash helpers/fp16-q8-control.sh > fp16-q8-control.log 2>&1
cat fp16-real-wire-oracle.log fp16-evidence.log
tail -n 5 fp16-q8-control.log
sha256sum fp16-evidence-50.tar.gz fp16-logits.tar.gz fp16-q8-control.tar.gz
