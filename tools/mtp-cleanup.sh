#!/usr/bin/env bash
set -euo pipefail
cd /home/xander
test -d /home/xander/hc-mtp
rm -rf -- /home/xander/hc-mtp
test ! -e /home/xander/hc-mtp
echo "MTP experiment directory removed"
