#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
export FG_SHADER_DIR="$PWD/source/vulkan"
[[ ! -e capture ]] || { echo "refusing stale native capture directory" >&2; exit 1; }
mkdir capture
tar -xf baseline-0.tar -C capture
rank="$(awk 'NR==1 {print $6}' capture/cases.tsv)"
[[ "$rank" =~ ^[1-7]$ ]]
tar -xf "baseline-$rank.tar" -C capture
sha256sum -c native-q8.gguf.sha256
source/tools/mtp-acceptance "$PWD/capture" "$PWD/capture/manifest.fgm" \
    "$PWD/native-q8.gguf" > capture/native.stdout 2> capture/native.stderr
python3 source/tools/mtp_capture.py summarize capture
sha256sum native-q8.gguf source/flash-gordon source/tools/mtp-acceptance \
    capture/manifest.fgm capture/embedding.q8 capture/head.q8 > capture/inputs.sha256
cp native-source.json capture/native-source.json
cd capture
tar -cf ../proposals.tar ./*.drafts.u32 acceptance.jsonl acceptance-summary.json \
    inputs.sha256 native-source.json native.stdout native.stderr
cd ..
sha256sum proposals.tar
