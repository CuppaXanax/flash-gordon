#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
tar -xf source.tar -C source
bash source/tools/mtp-small-build.sh
cd source
bash tools/mtp-small-isa.sh
bash tools/mtp-small-measure.sh
