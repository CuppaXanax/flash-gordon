#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
export TMPDIR="$PWD/.cache/compiler" XDG_CACHE_HOME="$PWD/.cache"
cp source/tools/mtp_source.py mtp_source.py
if ! .venv/bin/python -c 'import numpy' >/dev/null 2>&1; then
    bash source/tools/mtp-head-python.sh
fi
if [[ ! -f native-q8.gguf ]]; then
    bash source/tools/mtp-head-export.sh > capture-export.log 2>&1
fi
printf '%s  native-q8.gguf\n' 536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b \
    > native-q8.gguf.sha256
sha256sum -c native-q8.gguf.sha256
cd source
mkdir -p .build
make -j8 tools/bench-mtp-head > ../capture-gpu-build.log 2>&1
./tools/bench-mtp-head --self-test > ../capture-gpu-tests.jsonl
for script in tools/mtp-capture-*.sh tools/mtp-lease-guard.sh; do bash -n "$script"; done
sha256sum flash-gordon tools/mtp-acceptance vulkan/*.spv > release.sha256
tar -czf ../release.tar.gz flash-gordon tools/mtp-acceptance tools/mtp_capture.py vulkan release.sha256
cd ..
sha256sum source.tar release.tar.gz native-q8.gguf > prepared.sha256
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('.')
names=['source.tar','release.tar.gz','native-q8.gguf','source/flash-gordon','source/tools/mtp-acceptance']
records={n:hashlib.file_digest((p/n).open('rb'),'sha256').hexdigest() for n in names}
pathlib.Path('prepared.json').write_text(json.dumps(dict(
    kind='mtp-acceptance-prebuild',blade='192.168.42.51',production_fleet_accessed=False,
    files=records,corpus_minimum=dict(chat=20,code=20),tokens=259,anchors=256,
    greedy=dict(chat=4,code=4,k=[1,2,3],tokens=256,verifier='causal-N1-reference'),
    wall_budget_minutes=45,serving_small_n_linked=False),indent=2)+'\n')
PY
tar -czf prebuild-evidence.tar.gz capture-build.log capture-gpu-build.log \
    capture-core-tests.log capture-gpu-tests.jsonl capture-export.log prepared.json prepared.sha256 native-source.json
cat prepared.json
