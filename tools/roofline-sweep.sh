#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p sweep
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
tar -xf source.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk > ../sweep/build.txt 2>&1
cd ..
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > sweep/restored.txt
}
trap restore EXIT
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > sweep/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
/usr/local/bin/bc250-cu-live-manager --help > sweep/cu-help.txt 2>&1 || true
/usr/local/bin/bc250-cu-live-manager status > sweep/cu-status.txt 2>&1 || true
cd src
DS4_REMOTE_TEST_FILTER=roof_qsa ./tests/test_fg_vk > ../sweep/qsa-parity.txt 2>&1
for v in 1 2 3 4; do
    for mask in 2 66; do
        FG_ROOFLINE=$mask FG_ROOF_QSA=$v ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 \
            > "../sweep/qsa-$mask-$v.jsonl" 2> "../sweep/qsa-$mask-$v.log"
    done
done
for v in 1 2 3 4 5 6; do
    FG_ROOFLINE=1 FG_ROOF_DENSE=$v FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_COMPACT=1 \
        FG_BENCH_DENSE_SELECTED=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
        DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk > "../sweep/dense-$v.txt" 2>&1
done
for mask in 0 1 64 66 90 91; do
    FG_ROOFLINE=$mask ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 \
        > "../sweep/profile-$mask.jsonl" 2> "../sweep/profile-$mask.log"
done
cd ..
cat /sys/class/drm/card*/device/pp_dpm_sclk > sweep/sclk-end.txt
restore
trap - EXIT
tar -cf sweep.tar -C sweep .
cat sweep/restored.txt
