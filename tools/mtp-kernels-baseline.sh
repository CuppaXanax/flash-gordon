#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p src .build cache baseline/vulkan evidence
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk > ../baseline/build.txt 2>&1
cp tools/bench-decode-fusion ../baseline/bench
cp vulkan/*.spv ../baseline/vulkan/
cd ..
if [ -x /usr/local/bin/bc250-cu-live-manager ]; then
    /usr/local/bin/bc250-cu-live-manager status > evidence/cu-original.txt
else
    printf 'Live CU manager unavailable; driver count only\n' > evidence/cu-original.txt
    cat /sys/module/amdgpu/parameters/active_cu_number >> evidence/cu-original.txt 2>/dev/null ||
        printf 'Driver CU override not exposed\n' >> evidence/cu-original.txt
fi
cp /etc/cyan-skillfish-governor-smu/config.toml evidence/governor-original.toml
cat /sys/class/drm/card*/device/pp_dpm_sclk > evidence/sclk-original.txt
active=()
restore(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> evidence/services-restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        ready=0
        for attempt in $(seq 1 90); do
            host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
            if curl --max-time 3 -fsS "http://$host:$port/health" > "evidence/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf baseline.tar.gz evidence baseline/*.txt baseline/*.jsonl baseline/*.stderr
    exit "$rc"
}
trap restore EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
printf '%s\n' "${active[@]}" > evidence/services-original.txt
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
for round in 1 2 3 4 5; do
    for n in 1 2 3 4; do
        FG_ROOFLINE=0 FG_SMALL_KERNELS=0 FG_SHADER_DIR=../baseline/vulkan ../baseline/bench \
            --q4 --small-n --no-clocks --batch "$n" --seed "$round" \
            --warmup "$((256/n))" --windows 3 --iterations "$((64/n))" \
            > "../baseline/round-$round-n$n.jsonl" 2> "../baseline/round-$round-n$n.stderr"
    done
done
for n in 1 2 3 4; do
    FG_ROOFLINE=0 FG_SMALL_KERNELS=0 FG_SHADER_DIR=../baseline/vulkan ../baseline/bench \
        --q4 --small-n --no-clocks --batch "$n" --warmup "$((256/n))" \
        --iterations 1 --windows 1 --profile \
        > "../baseline/profile-n$n.jsonl" 2> "../baseline/profile-n$n.stderr"
done
cd ..
