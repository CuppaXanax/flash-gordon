#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
tar -xf q8-control.tar
printf '%s  %s\n' 918a6175bf6082863b2306c125802970164cbe9777847997e5c1c51aa3096684 original-dense-q8.gguf | sha256sum -c -
printf '%s  %s\n' 35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb original-q8.fgm | sha256sum -c -
[[ "$(sha256sum source/tools/bf16-offline | cut -d' ' -f1)" == 11ce46e6cb662c4d70509f28301c50a1bb3ff1dd9be49b1516cd9cd453f57239 ]]
rm q8-control.tar
sha256sum original-q8.fgm original-dense-q8.gguf hardware/tools/fp16_q8_restore.py
touch fp16-q8-source.ready
