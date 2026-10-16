#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
host=$(hostname -I)
case "$host" in
    *192.168.42.52*) rank=52;;
    *192.168.42.53*) rank=53;;
    *) exit 1;;
esac
{
    date -u --iso-8601=seconds
    hostname -I
    uname -a
    cc -dumpfullversion -dumpversion
    sha256sum source/tools/bench-bf16-hops source/tests/test_bf16_hops \
        source/src/protocol.c source/include/fg_protocol.h \
        /etc/cyan-skillfish-governor-smu/config.toml \
        /sys/class/drm/card*/device/power_dpm_force_performance_level
    for speed in /sys/class/net/enp*/speed; do
        if [[ -r "$speed" ]]; then echo "$speed $(cat "$speed") Mb/s"; fi
    done
} > pair-metadata.txt
tar -czf "evidence-$rank.tar.gz" pair-metadata.txt hop-*.jsonl hop-*.stderr \
    source/src/protocol.c source/include/fg_protocol.h source/tools/bench_bf16_hops.c
cat pair-metadata.txt
sha256sum "evidence-$rank.tar.gz"
