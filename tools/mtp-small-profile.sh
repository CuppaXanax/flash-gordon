#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-mtp/source
export XDG_CACHE_HOME="$PWD/.build/cache"
for n in 1 2 3 4; do
    ./tools/bench-decode-fusion --q4 --small-n --batch "$n" --profile \
        --warmup 12 --iterations 1 --windows 1 \
        > ".build/small-profile-${n}.jsonl" 2> ".build/small-profile-${n}.stderr"
    echo "== small-N $n"
    python3 -c 'import json,sys; a=[json.loads(x) for x in open(sys.argv[1])]; print(*[x for x in a if x["kind"] in ["sample","profile"]],sep="\n"); print(*sorted([x for x in a if x["kind"]=="kernel"],key=lambda x:-x["gpu_us"])[:16],sep="\n")' ".build/small-profile-${n}.jsonl"
done
./tools/bench-decode-fusion --q4 --batch 2 --profile \
    --warmup 12 --iterations 1 --windows 1 \
    > .build/legacy-profile-2.jsonl 2> .build/legacy-profile-2.stderr
echo "== legacy N=2"
python3 -c 'import json,sys; a=[json.loads(x) for x in open(sys.argv[1])]; print(*[x for x in a if x["kind"] in ["sample","profile"]],sep="\n"); print(*sorted([x for x in a if x["kind"]=="kernel"],key=lambda x:-x["gpu_us"])[:16],sep="\n")' .build/legacy-profile-2.jsonl
