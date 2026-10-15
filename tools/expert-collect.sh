#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
cd /home/xander/hc-experts
sha256sum baseline/*.spv vulkan/fg_moe_decode_gate_up.spv vulkan/fg_moe_decode_down_reduce.spv vulkan/fg_moe_decode_down_q8.spv
if test -r /home/xander/fg-ring-pack-abliterated/manifest.fgm; then
    printf 'Real model manifest available\n'
else
    printf 'Real model manifest unavailable on permitted blade\n'
fi
tar -czf evidence.tar.gz --exclude=evidence.tar.gz ./*.log ./*.bin ./*.bounds baseline/ isa.log.gz isa-baseline.log.gz
