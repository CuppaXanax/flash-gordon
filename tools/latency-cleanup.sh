#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
if pgrep -af '/home/xander/hc-lat/(latency-|bench-)|[.]?/tools/bench-ngram-latency|[.]?/bench-fabric-latency'; then
    echo "Refuse cleanup while benchmark processes remain" >&2
    exit 1
fi
systemctl is-active --quiet cyan-skillfish-governor-smu
if [ -f original.toml ]; then cmp original.toml /etc/cyan-skillfish-governor-smu/config.toml; fi
if firewall-cmd --query-port=19552/tcp >/dev/null; then exit 1; fi
if firewall-cmd --query-port=19600-19615/tcp >/dev/null; then exit 1; fi
if [ "$(hostname -I | grep -o '192.168.42.53')" = "192.168.42.53" ]; then
    test -d /home/xander/hc-q4
    echo "hc-q4 artifacts preserved"
fi
echo "Governor active, original config unchanged, benchmark firewall ports closed"
cd /home/xander
rm -rf -- /home/xander/hc-lat
test ! -e /home/xander/hc-lat
echo "hc-lat removed"
