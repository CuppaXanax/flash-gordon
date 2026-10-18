#!/bin/bash
set -euo pipefail
cd /home/xander/hc-fusion
systemctl is-active --quiet cyan-skillfish-governor-smu
cmp fixed/original.toml /etc/cyan-skillfish-governor-smu/config.toml
if pgrep -af '^./tools/bench-decode-fusion|^/usr/local/libexec/cyan-skillfish-governor-smu .*hc-fusion'; then
    echo "Experiment process still running; refusing cleanup." >&2
    exit 1
fi
cd /home/xander
rm -rf -- /home/xander/hc-fusion
test ! -e /home/xander/hc-fusion
systemctl is-active cyan-skillfish-governor-smu
echo "Fusion workspace removed; stock governor config unchanged."
