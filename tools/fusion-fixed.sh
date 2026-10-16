#!/bin/bash
set -euo pipefail
cd /home/xander/hc-fusion
mkdir -p src fixed .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk
for filter in decode_fusion_router decode_fusion_residual decode_fusion_shared decode_layer_clocks; do
    DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk
done > ../fixed/validation.txt 2>&1
cd ..
cp /etc/cyan-skillfish-governor-smu/config.toml fixed/original.toml
systemctl is-active --quiet cyan-skillfish-governor-smu
grep -q '^min = 1000$' fixed/original.toml
grep -q '^max = 2000$' fixed/original.toml
sed 's/^min = 1000$/min = 2000/' fixed/original.toml > fixed/pinned.toml
gp=
restore(){
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; fi
    systemctl start cyan-skillfish-governor-smu
    cmp fixed/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu > fixed/restored.txt
}
trap restore EXIT
systemctl stop cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/fixed/pinned.toml" > fixed/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cat /sys/class/drm/card*/device/pp_dpm_sclk > fixed/sclk-start.txt
cd src
for mask in 0 1 2 3 4 7; do
    FG_DECODE_FUSION=$mask ./tools/bench-decode-fusion --q4 --seed 5 --windows 3 \
        > "../fixed/ablation-$mask.jsonl" 2> "../fixed/ablation-$mask.log"
done
for round in 1 2 3; do
    for phase in a b a2; do
        mask=0
        if [ "$phase" = b ]; then mask=7; fi
        FG_DECODE_FUSION=$mask ./tools/bench-decode-fusion --q4 --seed "$round" \
            > "../fixed/q4-$round-$phase.jsonl" 2> "../fixed/q4-$round-$phase.log"
    done
done
for mask in 0 7; do
    FG_DECODE_FUSION=$mask ./tools/bench-decode-fusion --q4 --no-clocks \
        > "../fixed/no-clocks-$mask.jsonl" 2> "../fixed/no-clocks-$mask.log"
    FG_DECODE_FUSION=$mask ./tools/bench-decode-fusion --q4 --profile --warmup 256 \
        --iterations 1 --windows 1 > "../fixed/profile-$mask.jsonl" 2> "../fixed/profile-$mask.log"
    FG_DECODE_FUSION=$mask ./tools/bench-decode-fusion --windows 3 \
        > "../fixed/q8-$mask.jsonl" 2> "../fixed/q8-$mask.log"
done
sha256sum tools/bench-decode-fusion > ../fixed/executable-sha256.txt
cat /sys/class/drm/card*/device/pp_dpm_sclk > ../fixed/sclk-end.txt
cd ..
restore
trap - EXIT
tar -cf fixed.tar -C fixed .
cat fixed/sclk-start.txt fixed/sclk-end.txt fixed/restored.txt
