#!/usr/bin/env bash
set -euo pipefail
mkdir -p /home/xander/hc-lat
cd /home/xander/hc-lat
hostname
df -B1 /home/xander
lsblk -o NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS
free -b
command -v gcc glslangValidator fio || true
cat /sys/class/drm/card*/device/pp_dpm_sclk || true
ss -ltn | head -20
