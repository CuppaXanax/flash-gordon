#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
mkdir -p pack build cache source
if [[ -f metadata-source.tar.gz && ! -f metadata.ingested ]]; then
    tar -xzf metadata-source.tar.gz
    sha256sum -c metadata.sha256
    touch metadata.ingested
fi
if [[ -f rank-pack.sha256 ]]; then
    while read -r digest path; do
        [[ -f "$path" && ! -f "$path.verified" ]] || continue
        echo "$digest  $path" | sha256sum -c -
        touch "$path.verified"
    done < rank-pack.sha256
fi
df -B1 /home/xander | tail -n 1
