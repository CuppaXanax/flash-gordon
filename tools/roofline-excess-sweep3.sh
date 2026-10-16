#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence excess3 .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk tests/test_q4_repack > ../excess3/build.txt 2>&1
bash tools/check-flags.sh > ../excess3/flags.txt
./tests/test_q4_repack > ../excess3/repack.txt
cd ..
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > excess3/restored.txt
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "excess3/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ]
    done
}
finish(){
    rc=$?
    restore || rc=1
    tar -czf excess3.tar.gz --exclude=governor.log evidence excess excess2 excess3
    exit "$rc"
}
trap finish EXIT
/usr/local/bin/bc250-cu-live-manager status > excess3/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' excess3/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > excess3/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
for filter in roof_dense_exact roof_hc_down_inject roof_gdn_gate_controls; do
    DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk > "../excess3/$filter.txt" 2>&1
done
FG_ROOFLINE=4 FG_ROOF_RECURRENT=2 FG_BENCH_GDN_ALGEBRAIC=1 DS4_REMOTE_TEST_FILTER=gdn_algebraic \
    ./tests/test_fg_vk > ../excess3/recurrent-vec-oracle.txt 2>&1
capture(){
    name=$1 mask=$2 expert=$3 hc=$4 recurrent=$5 seed=$6
    FG_ROOFLINE=$mask FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert FG_ROOF_HC=$hc \
        FG_ROOF_RECURRENT=$recurrent ./tools/bench-decode-fusion --q4 --profile --warmup 256 \
        --iterations 1 --windows 1 --seed "$seed" > "../excess3/$name-$seed.jsonl" 2> "../excess3/$name-$seed.log"
}
for round in 1 2 3; do
    capture baseline 251 2 0 0 "$round"
    for hc in 0 4 8 12 16 20; do capture "hc-down-$hc" 763 2 "$hc" 0 "$round"; done
    capture hc-up 507 2 0 0 "$round"
    capture gdn-gate 1275 2 0 0 "$round"
    capture recurrent-vec 255 2 0 2 "$round"
    capture combination 2047 1024 0 2 "$round"
done
