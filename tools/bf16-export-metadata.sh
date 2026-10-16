#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
sha256sum pack/manifest.fgm pack/public-header.gguf pack/tokenizer/tokenizer.fgt \
    pack/source-map.json > metadata.sha256
tar -czf metadata-source.tar.gz metadata.sha256 tokenizer-source.json \
    pack/manifest.fgm pack/public-header.gguf pack/tokenizer/tokenizer.fgt pack/source-map.json
sha256sum metadata-source.tar.gz
