#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
hostname -I
df -h /home/xander
free -b
sha256sum /sys/class/drm/card*/device/power_dpm_force_performance_level \
    /sys/class/drm/card*/device/pp_power_profile_mode 2>/dev/null || true
for s in bc250-embed bc250-rerank; do systemctl is-active "$s" || true; done
find /home/xander -maxdepth 4 -type f \
    \( -name '*.fgw' -o -name '*.gguf' -o -name '*manifest*.json' \
       -o -name '*.fgm' -o -name '*corpus*.json' -o -name '*logit*' \) \
    -printf '%s %p\n' 2>/dev/null | sort
