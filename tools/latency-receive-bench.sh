#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
chmod +x bench-fabric-latency
pgrep -af 'flash-gordon|llama|bench|sol-|mtp' || true
file bench-fabric-latency
