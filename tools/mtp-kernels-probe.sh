#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
hostname
if [ -x /usr/local/bin/bc250-cu-live-manager ]; then
    /usr/local/bin/bc250-cu-live-manager status
else
    cat /sys/module/amdgpu/parameters/active_cu_number 2>/dev/null || true
fi
systemctl is-active cyan-skillfish-governor-smu || true
cat /sys/class/drm/card*/device/pp_dpm_sclk
echo 'Kernel workspace ready'
