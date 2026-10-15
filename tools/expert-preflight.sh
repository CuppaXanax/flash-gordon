#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
mkdir -p /home/xander/hc-experts
cd /home/xander/hc-experts
hostname
systemctl is-active bc250-embed bc250-rerank || true
free -m
for f in /sys/class/drm/card*/device/mem_info_vram_used /sys/class/drm/card*/device/mem_info_vram_total /sys/class/drm/card*/device/mem_info_gtt_used; do
    test ! -r "$f" || { printf '%s ' "$f"; cat "$f"; }
done
command -v cc
command -v glslangValidator || command -v glslc
if command -v vulkaninfo >/dev/null; then
    vulkaninfo 2>/dev/null | grep -E 'subgroupSize|activeComputeUnitCount|computeUnitsPerShaderArray' || true
fi
ss -ltnp
if test -r /sys/module/amdgpu/parameters/cu_mask; then
    cat /sys/module/amdgpu/parameters/cu_mask
fi
