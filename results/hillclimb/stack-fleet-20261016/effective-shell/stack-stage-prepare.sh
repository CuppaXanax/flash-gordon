#!/usr/bin/env bash
set -euo pipefail
root=/home/xander/hc-abq-stack-live-20261016
test ! -e "$root"
mkdir -m 700 "$root"
