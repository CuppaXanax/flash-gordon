#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat/src
mkdir -p .build
export TMPDIR=$PWD/.build XDG_CACHE_HOME=/home/xander/hc-lat/cache FG_SHADER_DIR=$PWD/vulkan
trap 'rm -f /home/xander/hc-lat/warm-fixture.iq4nl' EXIT
make -s -j8 tools/bench-ngram-latency
pgrep -af 'flash-gordon|llama|bench|sol-|mtp' || true
./tools/bench-ngram-latency /home/xander/hc-lat/warm-fixture.iq4nl --warm-only > ../ngram-warm.jsonl
cat ../ngram-warm.jsonl
