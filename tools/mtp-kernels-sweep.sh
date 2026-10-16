#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p sweep
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../sweep/build.txt 2>&1
cd ..
active=()
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> sweep/restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "sweep/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf sweep.tar.gz --exclude=sweep/governor.log sweep evidence
    exit "$rc"
}
trap finish EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
for service in "${active[@]}"; do systemctl stop "$service"; done
cat /sys/class/drm/card*/device/pp_dpm_sclk > sweep/sclk-observed.txt
cat /sys/class/drm/card*/device/pp_dpm_mclk > sweep/mclk-observed.txt 2>/dev/null || true
vulkaninfo > sweep/vulkaninfo.txt 2> sweep/vulkaninfo.stderr || true
cd src
for flags in 203 331; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --batch 2 \
        --no-clocks --warmup 1 --windows 1 --iterations 1 --small-test --rollback-test \
        --session-test --allow-reassociation > "../sweep/oracles-f$flags.jsonl" 2> "../sweep/oracles-f$flags.stderr"
done
for wg in 32 64 128 256; do
    FG_SMALL_KERNELS=331 ./tools/bench-decode-fusion --q4 --no-clocks --dense-only \
        --dense-type 12 --dense-shape 2560 10240 --small-dense-wg "$wg" --allow-reassociation \
        > "../sweep/dense-wg$wg.jsonl" 2> "../sweep/dense-wg$wg.stderr"
done
for flags in 0 75 203 331; do
    for n in 2 3 4; do
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
            --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
            > "../sweep/profile-f$flags-n$n.jsonl" 2> "../sweep/profile-f$flags-n$n.stderr"
    done
done
RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=331 \
    ./tools/bench-decode-fusion --q4 --small-n --no-clocks --batch 4 \
    --warmup 2 --iterations 1 --windows 1 > ../sweep/isa.jsonl 2> ../sweep/isa.stderr
cd ..
