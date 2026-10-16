#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof/src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan XDG_CACHE_HOME=/home/xander/hc-roof/cache
make -s -j8 tests/test_fg_vk > ../incremental/rejected-q8-rows-build.txt 2>&1
sha256sum tools/bench-decode-fusion tests/test_fg_vk ../source.tar > ../incremental/binary-source-sha256.txt
restore(){
    local status=0
    systemctl start bc250-embed bc250-rerank || status=1
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "../incremental/extra-health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || status=1
    done
    return "$status"
}
finish(){
    rc=$?
    restore || rc=1
    cd ..
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank || rc=1
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    /usr/local/bin/bc250-cu-live-manager status | grep 'CUs active & routed  : 40/40' || rc=1
    tar -czf excess-final.tar.gz --exclude='*.f32' --exclude='*.gdn' --exclude=governor.log \
        evidence excess excess2 excess3 excess4 excess5 incremental cumulative
    exit "$rc"
}
trap finish EXIT
systemctl stop bc250-embed bc250-rerank
FG_ROOFLINE=0 FG_ROOF_HC=2 DS4_REMOTE_TEST_FILTER=q8_decode_shape_parity ./tests/test_fg_vk \
    > ../incremental/rejected-q8-rows-oracle.txt 2>&1
for v in 23 24; do
    FG_ROOFLINE=1 FG_ROOF_DENSE=$v FG_BENCH_DENSE_QUANT=1 FG_BENCH_DENSE_COMPACT=1 \
        FG_BENCH_DENSE_SELECTED=1 FG_BENCH_IQ_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
        DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk > "../incremental/rejected-iq4-$v-oracle.txt" 2>&1
done
for v in 6 7; do
    FG_ROOFLINE=32 FG_ROOF_EXPERT=$v FG_BENCH_EXPERT_DECODE=80 FG_BENCH_EXPERT_FORMAT=q4q5 \
        DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark ./tests/test_fg_vk > "../incremental/rejected-expert-$v-oracle.txt" 2>&1
done
