#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence excess4 .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../excess4/build.txt 2>&1
cd ..
gp= restricted=0
restore(){
    cd /home/xander/hc-roof
    local status=0
    if [ "$restricted" = 1 ]; then
        /usr/local/bin/bc250-cu-live-manager --yes apply-service > excess4/cu-restored-apply.txt 2>&1 || status=1
        restricted=0
    fi
    /usr/local/bin/bc250-cu-live-manager status > excess4/cu-restored-status.txt
    grep -q 'CUs active & routed  : 40/40' excess4/cu-restored-status.txt || status=1
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank || status=1
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml || status=1
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > excess4/restored.txt || status=1
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "excess4/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || status=1
    done
    return "$status"
}
finish(){
    rc=$?
    restore || rc=1
    tar -czf excess4.tar.gz --exclude=governor.log evidence excess excess2 excess3 excess4
    exit "$rc"
}
trap finish EXIT
/usr/local/bin/bc250-cu-live-manager status > excess4/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' excess4/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > excess4/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
capture(){
    label=$1 expert=$2 seed=$3
    FG_ROOFLINE=251 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert FG_ROOF_HC=0 \
        FG_ROOF_RECURRENT=0 ./tools/bench-decode-fusion --q4 --profile --warmup 256 \
        --iterations 1 --windows 1 --seed "$seed" > "../excess4/cu$cu-$label-$seed.jsonl" 2> "../excess4/cu$cu-$label-$seed.log"
}
for cu in 40 24; do
    if [ "$cu" = 24 ]; then
        restricted=1
        /usr/local/bin/bc250-cu-live-manager --yes stock-dispatch > ../excess4/cu24-apply.txt
        /usr/local/bin/bc250-cu-live-manager status > ../excess4/cu24-status.txt
        grep -q 'CUs active & routed  : 24/40' ../excess4/cu24-status.txt
    fi
    for v in 8194 16386 32770 65538; do
        FG_ROOFLINE=32 FG_ROOF_EXPERT=$v FG_BENCH_EXPERT_DECODE=80 FG_BENCH_EXPERT_FORMAT=q4q5 \
            DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark ./tests/test_fg_vk > "../excess4/cu$cu-expert-$v-shape.txt" 2>&1
    done
    for round in 1 2 3; do
        capture baseline 2 "$round"
        for v in 8194 16386 32770 65538 9216 17408 33792 66560; do capture "expert-$v" "$v" "$round"; done
    done
done
