#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
systemctl list-units --type=service --all --no-pager | grep -E 'bc250|cyan' || true
systemctl cat cyan-skillfish-governor-smu
cat /sys/class/drm/card*/device/pp_dpm_sclk
cat /sys/class/drm/card*/device/pp_dpm_mclk
for f in /sys/kernel/debug/dri/*/amdgpu_gca_config; do echo "$f"; od -An -tx4 "$f"; done
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null || true
