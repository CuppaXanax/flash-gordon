#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
{
    sha256sum src/src/ngram.c src/src/runtime.c src/tests/test_fg_vk.c \
        src/tools/bench_ngram_latency.c src/tools/bench_fabric_latency.c
    stat -f -c 'filesystem=%T' .
    df -B1 .
    systemctl is-active cyan-skillfish-governor-smu
    cmp original.toml /etc/cyan-skillfish-governor-smu/config.toml
    test -d /home/xander/hc-q4
} > final-state.txt
tar -czf captures.tar.gz final-state.txt final-build.log \
    validation*.txt ngram-store.jsonl ngram-store-pagecache-only.jsonl \
    ngram-warm.jsonl storage53.jsonl original.toml pinned.toml \
    sclk-start.txt sclk-end.txt restored.txt
cat final-state.txt
