#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
mkdir -p build cache
export TMPDIR=$PWD/build XDG_CACHE_HOME=$PWD/cache
pgrep -af 'flash-gordon|llama|bench|sol-|mtp' || true
if [ -f ngram-store.jsonl ]; then mv ngram-store.jsonl ngram-store-pagecache-only.jsonl; fi
gcc -O2 -std=c17 -Wall -Wextra -Werror src/tools/latency-storage.c -pthread -o latency-storage
trap 'rm -f /home/xander/hc-lat/synthetic.iq4nl' EXIT
./latency-storage /home/xander/hc-lat/synthetic.iq4nl keep > storage53.jsonl
cd src
export FG_SHADER_DIR=$PWD/vulkan
./tools/bench-ngram-latency /home/xander/hc-lat/synthetic.iq4nl > ../ngram-store.jsonl 2> ../ngram-store.log
cat ../ngram-store.jsonl
