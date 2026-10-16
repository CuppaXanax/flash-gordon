#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p prequal
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
tar -xf source.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tools/bench-decode-fusion tests/test_fg_vk > ../prequal/build.txt 2>&1
bash tools/check-flags.sh > ../prequal/flags.txt
cd ..
gp=
restore(){
    cd /home/xander/hc-roof
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > prequal/restored.txt
}
trap restore EXIT
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > prequal/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
for filter in roof_controls roof_qsa hc_inject_partial gr_partial_boundaries decode_fusion_router expert_decode_fused_random gdn_algebraic; do
    FG_ROOFLINE=255 DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk > "../prequal/test-$filter.txt" 2>&1
done
for mask in 0 8 16 128 91 219; do
    FG_ROOFLINE=$mask FG_ROOF_QSA=3 ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 \
        > "../prequal/profile-$mask.jsonl" 2> "../prequal/profile-$mask.log"
done
cd ..
cat /sys/class/drm/card*/device/pp_dpm_sclk > prequal/sclk-end.txt
restore
trap - EXIT
tar -cf prequal.tar -C prequal .
cat prequal/restored.txt
