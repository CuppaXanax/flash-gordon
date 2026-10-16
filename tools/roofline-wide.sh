#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../bandwidth/wide-build.txt 2>&1
cd ..
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > bandwidth/wide-restored.txt
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "bandwidth/wide-health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ]
    done
}
trap restore EXIT
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > bandwidth/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
FG_ROOFLINE=1 FG_ROOF_DENSE=21 FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_SOA=1 \
    FG_BENCH_DENSE_SELECTED=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
    DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk > ../bandwidth/dense-21.txt 2>&1
FG_ROOFLINE=1 FG_ROOF_DENSE=22 FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_COMPACT=1 \
    FG_BENCH_DENSE_SELECTED=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
    DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk > ../bandwidth/dense-22.txt 2>&1
for round in 1 2 3; do
    for variant in 19 21 22; do
        FG_ROOFLINE=1 FG_ROOF_DENSE=$variant ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
            > "../bandwidth/dense-$variant-$round.jsonl" 2> "../bandwidth/dense-$variant-$round.log"
    done
done
cd ..
restore
trap - EXIT
tar -cf bandwidth.tar --exclude=governor.log evidence bandwidth
cat bandwidth/wide-restored.txt
