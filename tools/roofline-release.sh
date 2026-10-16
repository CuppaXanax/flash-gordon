#!/bin/bash
set -euo pipefail
shopt -s nullglob
cd /home/xander/hc-roof
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
tar -xf source.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk > ../final/release-build.txt 2>&1
bash tools/check-flags.sh > ../final/release-flags.txt
DS4_REMOTE_TEST_FILTER=roof_dense_exact ./tests/test_fg_vk > ../final/test-roof-dense-exact.txt 2>&1
cd ..
python3 - <<'PY' > final/rejected-errors.json
import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location("compare","src/tools/roofline-compare.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
results={}
for directory in sorted(Path(".").glob("prior-final-*")):
    for candidate in directory.glob("*-b.f32"):
        reference=Path(str(candidate).replace("-b.f32","-a.f32"))
        config=next(json.loads(line) for line in candidate.with_suffix(".jsonl").read_text().splitlines()
                    if json.loads(line)["kind"]=="roofline")
        results[str(candidate)]={"flags":config,"output":mod.compare(reference,candidate,1.0,True),
                                "state":mod.compare(str(reference)+".gdn",str(candidate)+".gdn",1.0,False)}
print(json.dumps(results,indent=2))
PY
systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > final/release-services.txt
cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
/usr/local/bin/bc250-cu-live-manager status > final/release-cus.txt
grep -q 'CUs active & routed  : 40/40' final/release-cus.txt
curl --max-time 10 -fsS http://192.168.42.50:8091/health > final/release-health-8091.json
curl --max-time 10 -fsS http://192.168.42.50:8092/health > final/release-health-8092.json
tar -czf captures.tar.gz --exclude='*.f32' --exclude='*.gdn' --exclude=governor.log --exclude=services.txt \
    evidence sweep prequal numerics final prior-final-*
cat final/test-roof-dense-exact.txt final/release-services.txt final/release-health-8091.json final/release-health-8092.json
