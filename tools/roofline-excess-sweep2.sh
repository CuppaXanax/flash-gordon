#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence excess2 .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk tests/test_q4_repack > ../excess2/build.txt 2>&1
./tests/test_q4_repack > ../excess2/repack.txt
cd ..
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > excess2/restored.txt
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "excess2/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ]
    done
}
finish(){
    rc=$?
    restore || rc=1
    tar -czf excess2.tar.gz --exclude=governor.log evidence excess excess2
    exit "$rc"
}
trap finish EXIT
/usr/local/bin/bc250-cu-live-manager status > excess2/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' excess2/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > excess2/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
FG_ROOFLINE=256 DS4_REMOTE_TEST_FILTER=roof_hc_up_mix ./tests/test_fg_vk > ../excess2/hc-oracle.txt 2>&1
FG_ROOFLINE=4 FG_ROOF_RECURRENT=1 FG_BENCH_GDN_ALGEBRAIC=1 DS4_REMOTE_TEST_FILTER=gdn_algebraic \
    ./tests/test_fg_vk > ../excess2/recurrent-oracle.txt 2>&1
for v in 23 24 25 26 27 28; do
    FG_ROOFLINE=1 FG_ROOF_DENSE=$v FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_COMPACT=1 \
        FG_BENCH_DENSE_SELECTED=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
        DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk > "../excess2/dense-$v.txt" 2>&1
done
for v in 2 64 192 256 512 1024 2050 4098 2112 4160; do
    FG_ROOFLINE=32 FG_ROOF_EXPERT=$v FG_BENCH_EXPERT_DECODE=80 FG_BENCH_EXPERT_FORMAT=q4q5 \
        DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark ./tests/test_fg_vk > "../excess2/expert-$v-shape.txt" 2>&1
done
capture(){
    name=$1 mask=$2 dense=$3 expert=$4 hc=$5 recurrent=$6 seed=$7
    FG_ROOFLINE=$mask FG_ROOF_DENSE=$dense FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert FG_ROOF_HC=$hc \
        FG_ROOF_RECURRENT=$recurrent ./tools/bench-decode-fusion --q4 --profile --warmup 256 \
        --iterations 1 --windows 1 --seed "$seed" > "../excess2/$name-$seed.jsonl" 2> "../excess2/$name-$seed.log"
}
for round in 1 2 3; do
    capture baseline 0 19 2 0 0 "$round"
    for v in 23 24 25 26 27 28; do capture "dense-$v" 1 "$v" 2 0 0 "$round"; done
    for v in 64 192 256 512 1024 2050 4098 2112 4160; do capture "expert-$v" 32 19 "$v" 0 0 "$round"; done
    capture hc-fuse 256 19 2 0 0 "$round"
    capture hc-expand 0 19 2 1 0 "$round"
    capture hc-rows 0 19 2 2 0 "$round"
    capture hc-q8-fuse 256 19 2 1 0 "$round"
    capture recurrent 4 19 2 0 1 "$round"
done
