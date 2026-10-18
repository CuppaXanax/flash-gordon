#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence bandwidth .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../bandwidth/build.txt 2>&1
bash tools/check-flags.sh > ../bandwidth/flags.txt
cd ..
cp /etc/cyan-skillfish-governor-smu/config.toml evidence/original.toml
sed 's/^min = 1000$/min = 2000/' evidence/original.toml > evidence/pinned.toml
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > bandwidth/restored.txt
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "bandwidth/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ]
    done
}
trap restore EXIT
/usr/local/bin/bc250-cu-live-manager status > bandwidth/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' bandwidth/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > bandwidth/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cat /sys/class/drm/card*/device/pp_dpm_sclk > bandwidth/sclk-start.txt
cd src
for v in 6 8 9 10 11 12 13 14 15 16 17 18 20; do
    FG_ROOFLINE=1 FG_ROOF_DENSE=$v FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_COMPACT=1 \
        FG_BENCH_DENSE_SELECTED=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
        DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk > "../bandwidth/dense-$v.txt" 2>&1
done
for v in 0 1 2 3 4 5 8 12; do
    FG_ROOFLINE=32 FG_ROOF_EXPERT=$v DS4_REMOTE_TEST_FILTER=expert_decode_fused_random \
        ./tests/test_fg_vk > "../bandwidth/expert-$v.txt" 2>&1
done
for round in 1 2 3; do
    FG_ROOFLINE=0 ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
        > "../bandwidth/baseline-$round.jsonl" 2> "../bandwidth/baseline-$round.log"
    for v in 6 8 9 10 11 12 13 14 15 16 17 18 20; do
        FG_ROOFLINE=1 FG_ROOF_DENSE=$v ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
            > "../bandwidth/dense-$v-$round.jsonl" 2> "../bandwidth/dense-$v-$round.log"
    done
    for v in 0 1 2 3 4 5 8 12; do
        FG_ROOFLINE=32 FG_ROOF_EXPERT=$v ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
            > "../bandwidth/expert-$v-$round.jsonl" 2> "../bandwidth/expert-$v-$round.log"
    done
    FG_ROOFLINE=219 FG_ROOF_DENSE=6 FG_ROOF_QSA=3 ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
        > "../bandwidth/readmit-$round.jsonl" 2> "../bandwidth/readmit-$round.log"
done
cd ..
cat /sys/class/drm/card*/device/pp_dpm_sclk > bandwidth/sclk-end.txt
restore
trap - EXIT
tar -cf bandwidth.tar --exclude=governor.log evidence bandwidth
cat bandwidth/restored.txt bandwidth/health-8091.json bandwidth/health-8092.json
