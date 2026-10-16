#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
for i in $(seq 1 120); do
    [[ ! -f f32-head.sha256 ]] || break
    [[ $i -lt 120 ]] || exit 1
    sleep 5
done
tar -czf fp16-baseline.tar.gz f32-quality.jsonl f32-stage-*.jsonl logits/f32
sha256sum fp16-baseline.tar.gz
