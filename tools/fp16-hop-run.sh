#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
export FG_FP16_SCALAR=0
tr -d '\r' < source/tools/bf16-hop-run.sh > build/hop-run.sh
for codec in fp16 fp16-128; do
    HOP_BENCH_CODEC="$codec" bash build/hop-run.sh > "$codec-transport.jsonl"
    cat "$codec-transport.jsonl"
done
