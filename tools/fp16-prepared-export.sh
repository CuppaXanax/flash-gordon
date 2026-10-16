#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
test -f fp16-q8.complete
cmp -s pack/input.teacher pack/input.candidate
sha256sum pack/cases.bin pack/ngram.f32 pack/input.teacher pack/manifest.fgm \
    original-q8.fgm prompts.fgev > fp16-prepared80.sha256
tar -czf fp16-prepared80.tar.gz pack/cases.bin pack/ngram.f32 pack/input.teacher \
    pack/manifest.fgm original-q8.fgm prompts.fgev fp16-prepared80.sha256
sha256sum fp16-prepared80.tar.gz
