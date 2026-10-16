#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p round8
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "round8/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in round8/*.txt round8/*.stderr; do echo "=== $f"; tail -5 "$f"; done
    fi
    tar -czf round8.tar.gz round8 evidence
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
sha256sum vulkan/*.spv > ../round8/spirv-before-sha256.txt
cd ..
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan FG_EXPERT_Q8_WIDE=1
export FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024
export FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 FG_NGRAM_LOCAL_CACHE=1 FG_SUBMIT_GROUP=2
unset FG_TEST_Q8_DOWN_TENSOR
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk > ../round8/build.txt 2>&1
sha256sum tools/bench-decode-fusion tests/test_fg_vk > ../round8/binary-sha256.txt
sha256sum vulkan/*.spv > ../round8/spirv-sha256.txt
cmp ../round8/spirv-before-sha256.txt ../round8/spirv-sha256.txt
FG_SMALL_KERNELS=2097482 DS4_REMOTE_TEST_FILTER=small_dense_vector_layout \
    ./tests/test_fg_vk > ../round8/vector-residency.txt 2>&1
FG_SMALL_KERNELS=8388608 DS4_REMOTE_TEST_FILTER=small_qsa_select \
    ./tests/test_fg_vk > ../round8/wave-topk-cpu.txt 2>&1
for wide in 0 1; do
    FG_EXPERT_Q8_WIDE=$wide FG_SMALL_KERNELS=4194634 DS4_REMOTE_TEST_FILTER=small_q8_down_native \
        ./tests/test_fg_vk > "../round8/exact-q8-wide$wide.txt" 2>&1
done
for flags in 0 330 14684490; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --mixed-experts --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../round8/mixed-oracles-f$flags.jsonl" 2> "../round8/mixed-oracles-f$flags.stderr"
done
FG_SMALL_KERNELS=14684490 ./tools/bench-decode-fusion --q4 --mixed-experts --small-n --no-clocks \
    --batch 4 --warmup 1024 --iterations 1 --windows 1 --small-test \
    --allow-reassociation > ../round8/long-oracles.jsonl 2> ../round8/long-oracles.stderr
