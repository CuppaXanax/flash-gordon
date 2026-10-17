#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-reconcile
tar -xf controls.tar
test -f expected-prefix80.json
test -f expected-ranks.txt
