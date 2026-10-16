#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
df -h .
pgrep -af 'bf16_source.py|bf16-offline' || true
for name in ngram-source.log rank0-source.log offline.log; do
    if [[ -f "$name" ]]; then echo "=== $name"; tail -n 4 "$name"; fi
done
for name in prepare.jsonl prepare.stderr stage-*.jsonl stage-*.stderr quality.jsonl quality.stderr; do
    if [[ -f "$name" ]]; then echo "=== $name"; tail -n 2 "$name"; fi
done
if [[ -d pack ]]; then
    find pack -maxdepth 1 -type f -printf '%s %f\n' | sort
fi
