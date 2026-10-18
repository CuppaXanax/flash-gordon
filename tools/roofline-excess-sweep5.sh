#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence excess5 .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../excess5/build.txt 2>&1
bash tools/check-flags.sh > ../excess5/flags.txt
cd ..
gp= restricted=0
restore(){
    cd /home/xander/hc-roof
    local status=0
    if [ "$restricted" = 1 ]; then
        /usr/local/bin/bc250-cu-live-manager --yes apply-service > excess5/cu-restored-apply.txt 2>&1 || status=1
        restricted=0
    fi
    /usr/local/bin/bc250-cu-live-manager status > excess5/cu-restored-status.txt
    grep -q 'CUs active & routed  : 40/40' excess5/cu-restored-status.txt || status=1
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank || status=1
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml || status=1
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > excess5/restored.txt || status=1
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "excess5/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || status=1
    done
    return "$status"
}
finish(){
    rc=$?
    restore || rc=1
    tar -czf excess5.tar.gz --exclude=governor.log evidence excess excess2 excess3 excess4 excess5
    exit "$rc"
}
trap finish EXIT
/usr/local/bin/bc250-cu-live-manager status > excess5/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' excess5/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > excess5/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
capture(){
    label=$1 mask=$2 hc=$3 gdn=$4 recurrent=$5 seed=$6 expert=${7:-1024}
    FG_ROOFLINE=$mask FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert FG_ROOF_HC=$hc \
        FG_ROOF_GDN=$gdn FG_ROOF_RECURRENT=$recurrent ./tools/bench-decode-fusion --q4 --profile --warmup 256 \
        --iterations 1 --windows 1 --seed "$seed" > "../excess5/cu$cu-$label-$seed.jsonl" 2> "../excess5/cu$cu-$label-$seed.log"
}
for cu in 40 24; do
    if [ "$cu" = 24 ]; then
        restricted=1
        /usr/local/bin/bc250-cu-live-manager --yes stock-dispatch > ../excess5/cu24-apply.txt
        /usr/local/bin/bc250-cu-live-manager status > ../excess5/cu24-status.txt
        grep -q 'CUs active & routed  : 24/40' ../excess5/cu24-status.txt
    fi
    DS4_REMOTE_TEST_FILTER=roof_dense_exact ./tests/test_fg_vk > "../excess5/cu$cu-dense-exact.txt" 2>&1
    DS4_REMOTE_TEST_FILTER=roof_hc_down_inject ./tests/test_fg_vk > "../excess5/cu$cu-hc-down-oracle.txt" 2>&1
    for gdn in 1 2 4; do
        FG_ROOF_GDN=$gdn DS4_REMOTE_TEST_FILTER=roof_gdn_gate_controls ./tests/test_fg_vk > "../excess5/cu$cu-gdn-$gdn-oracle.txt" 2>&1
    done
    FG_ROOFLINE=4 FG_ROOF_RECURRENT=3 FG_BENCH_GDN_ALGEBRAIC=1 DS4_REMOTE_TEST_FILTER=gdn_algebraic \
        ./tests/test_fg_vk > "../excess5/cu$cu-recurrent-single-oracle.txt" 2>&1
    for round in 1 2 3; do
        capture baseline 251 0 8 0 "$round" 2
        for expert in 64 256 512; do capture "expert-$expert" 251 0 8 0 "$round" "$expert"; done
        for hc in 32 40 48 96 160; do capture "hc-down-$hc" 763 "$hc" 8 0 "$round"; done
        for gdn in 1 2 4; do capture "gdn-$gdn" 1275 0 "$gdn" 0 "$round"; done
        capture hc-up 507 0 8 0 "$round"
        capture recurrent 255 0 8 1 "$round"
        capture recurrent-single 255 0 8 3 "$round"
    done
done
