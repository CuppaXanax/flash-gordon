#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname -I)" == *192.168.42.50* ]]
root=/home/xander/hc-reconcile
mkdir -p "$root"/{build,cache,pack,source,evidence}
df -B1 /home/xander | tail -n 1
free -b
for service in bc250-embed bc250-rerank; do
    printf '%s ' "$service"
    systemctl is-active "$service" || true
done
sha256sum /etc/cyan-skillfish-governor-smu/config.toml
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
