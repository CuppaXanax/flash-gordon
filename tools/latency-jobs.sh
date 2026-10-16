#!/usr/bin/env bash
set -euo pipefail
pgrep -af 'flash-gordon|llama|bench|sol-|mtp|python' || true
df -B1 /home/xander
stat -f -c 'filesystem=%T' /home/xander/hc-lat
if [ -e /home/xander/hc-lat/synthetic.iq4nl ]; then
    stat -c 'synthetic_bytes=%s allocated_blocks=%b block_bytes=%B' /home/xander/hc-lat/synthetic.iq4nl
fi
ls -lh /home/xander/hc-lat/*.jsonl 2>/dev/null || true
