#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p numerics
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > numerics/restored.txt
}
trap restore EXIT
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > numerics/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
for mask in 0 1 8 16 64 128 91; do
    FG_ROOFLINE=$mask FG_ROOF_DENSE=6 FG_ROOF_QSA=3 ./tools/bench-decode-fusion --q4 --seed 1 \
        --warmup 256 --iterations 64 --windows 5 --dump "../numerics/$mask.f32" \
        > "../numerics/$mask.jsonl" 2> "../numerics/$mask.log"
done
python3 - <<'PY' > ../numerics/errors.json
import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location("compare","tools/roofline-compare.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
results={}
for mask in (1,8,16,64,128,91,219):
    p=Path("../numerics") if mask!=219 else Path("../final")
    a=p/("0.f32" if mask!=219 else "cu40-1-a.f32")
    b=p/(f"{mask}.f32" if mask!=219 else "cu40-1-b.f32")
    results[mask]={"output":mod.compare(a,b,1.0,True),
                   "state":mod.compare(str(a)+".gdn",str(b)+".gdn",1.0,False)}
print(json.dumps(results,indent=2))
PY
cd ..
restore
trap - EXIT
tar -cf numerics.tar -C numerics --exclude=*.f32 --exclude=*.gdn .
cat numerics/errors.json numerics/restored.txt
