#!/usr/bin/env bash
set -euo pipefail
mkdir -p /home/xander/hc-mtp
cd /home/xander/hc-mtp
printf 'host '; hostname
printf 'memory '; free -h
printf 'disk '; df -h .
command -v cc make glslc python3
python3 -c 'import numpy; print("numpy", numpy.__version__)' || true
find /home/xander -maxdepth 3 -name flash-gordon -type f -print | head -10
ls /usr/include/vulkan/vulkan.h
