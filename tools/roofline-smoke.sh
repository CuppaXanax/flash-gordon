#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk > ../evidence/build.txt 2>&1
bash tools/check-flags.sh > ../evidence/flags.txt
cd ..
cp /etc/cyan-skillfish-governor-smu/config.toml evidence/original.toml
sed 's/^min = 1000$/min = 2000/' evidence/original.toml > evidence/pinned.toml
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > evidence/restored.txt
}
trap restore EXIT
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > evidence/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cat /sys/class/drm/card*/device/pp_dpm_sclk > evidence/sclk-start.txt
systemctl cat bc250-cu-live-manager > evidence/cu-manager.txt
vulkaninfo 2>/dev/null | grep -E -A6 'ShaderCoreProperties|activeCompute|shaderEngineCount|computeUnitsPerShaderArray' > evidence/vulkan-cus.txt || true
cd src
for filter in dense_quant_compact decode_fusion_router gdn_algebraic hc_inject_partial expert_decode_fused_random; do
    FG_ROOFLINE=63 DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk > "../evidence/test-$filter.txt" 2>&1
done
for v in 1 2 3 4 5; do
    FG_ROOFLINE=1 FG_ROOF_DENSE=$v ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 \
        > "../evidence/dense-$v.jsonl" 2> "../evidence/dense-$v.log"
done
for mask in 0 2 4 8 16 32 63; do
    FG_ROOFLINE=$mask ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 \
        > "../evidence/profile-$mask.jsonl" 2> "../evidence/profile-$mask.log"
done
cd ..
cat /sys/class/drm/card*/device/pp_dpm_sclk > evidence/sclk-end.txt
restore
trap - EXIT
tar -cf evidence.tar -C evidence .
cat evidence/restored.txt
