#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
case "$(hostname -I)" in
    *192.168.42.52*) rank=52;;
    *192.168.42.53*) rank=53;;
    *) exit 1;;
esac
{
    date -u --iso-8601=seconds
    uname -a
    cc -dumpfullversion -dumpversion
    sha256sum source/tools/bench-bf16-hops source/tests/test_bf16_hops \
        source/src/protocol.c source/include/fg_protocol.h \
        /etc/cyan-skillfish-governor-smu/config.toml \
        /sys/class/drm/card*/device/power_dpm_force_performance_level
    for speed in /sys/class/net/enp*/speed; do
        if [[ -r "$speed" ]]; then echo "$speed $(cat "$speed") Mb/s"; fi
    done
} > fp16-pair-metadata.txt
lscpu -J > fp16-cpu.json
tar -czf "fp16-evidence-$rank.tar.gz" fp16-pair-metadata.txt fp16-cpu.json \
    fp16-transport.jsonl fp16-128-transport.jsonl hop-*.stderr \
    source/src/protocol.c source/include/fg_protocol.h source/tools/bench_bf16_hops.c
cat fp16-pair-metadata.txt
sha256sum "fp16-evidence-$rank.tar.gz"
