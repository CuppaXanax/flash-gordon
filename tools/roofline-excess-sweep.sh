#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence excess .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../excess/build.txt 2>&1
cd ..
cp /etc/cyan-skillfish-governor-smu/config.toml evidence/original.toml
sed 's/^min = 1000$/min = 2000/' evidence/original.toml > evidence/pinned.toml
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > excess/restored.txt
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "excess/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ]
    done
}
finish(){
    rc=$?
    restore || rc=1
    tar -czf excess.tar.gz --exclude=governor.log evidence excess
    exit "$rc"
}
trap finish EXIT
/usr/local/bin/bc250-cu-live-manager status > excess/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' excess/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > excess/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
for v in 0 2 6 7 18 34 50 64 128; do
    FG_ROOFLINE=32 FG_ROOF_EXPERT=$v DS4_REMOTE_TEST_FILTER=expert_decode_fused_random \
        ./tests/test_fg_vk > "../excess/expert-$v-random.txt" 2>&1
    FG_ROOFLINE=32 FG_ROOF_EXPERT=$v FG_BENCH_EXPERT_DECODE=80 FG_BENCH_EXPERT_FORMAT=q4q5 \
        DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark ./tests/test_fg_vk > "../excess/expert-$v-shape.txt" 2>&1
done
for round in 1 2 3; do
    for v in 0 2 6 7 18 34 50 64 128; do
        FG_ROOFLINE=32 FG_ROOF_EXPERT=$v ./tools/bench-decode-fusion \
            --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
            > "../excess/expert-$v-$round.jsonl" 2> "../excess/expert-$v-$round.log"
    done
done
