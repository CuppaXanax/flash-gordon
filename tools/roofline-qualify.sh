#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
if [ -d final ]; then mv final "prior-final-$(date +%s)"; fi
mkdir -p final
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
tar -xf source.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../final/build.txt 2>&1
bash tools/check-flags.sh > ../final/flags.txt
cd ..
gp= restricted=0
restore(){
    cd /home/xander/hc-roof
    local status=0
    if [ "$restricted" = 1 ]; then
        /usr/local/bin/bc250-cu-live-manager --yes apply-service > final/cu-restored-apply.txt 2>&1 || status=1
        restricted=0
    fi
    /usr/local/bin/bc250-cu-live-manager status > final/cu-restored-status.txt 2>&1 || status=1
    grep -q 'CUs active & routed  : 40/40' final/cu-restored-status.txt || status=1
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank || status=1
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml || status=1
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > final/restored.txt || status=1
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "final/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || status=1
    done
    return "$status"
}
trap restore EXIT
/usr/local/bin/bc250-cu-live-manager status > final/cu-original-status.txt 2>&1
grep -q 'CUs active & routed  : 40/40' final/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > final/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cat /sys/class/drm/card*/device/pp_dpm_sclk > final/sclk-start.txt
cd src
for cu in 40 24; do
    if [ "$cu" = 24 ]; then
        restricted=1
        /usr/local/bin/bc250-cu-live-manager --yes stock-dispatch > ../final/cu24-apply.txt 2>&1
        /usr/local/bin/bc250-cu-live-manager status > ../final/cu24-status.txt 2>&1
        grep -q 'CUs active & routed  : 24/40' ../final/cu24-status.txt
    fi
    for round in 1 2 3 4 5; do
        for phase in a b a2; do
            mask=0
            if [ "$phase" = b ]; then mask=27; fi
            FG_ROOFLINE=$mask FG_ROOF_DENSE=7 FG_ROOF_QSA=3 ./tools/bench-decode-fusion \
                --q4 --seed "$round" --windows 5 --iterations 64 --warmup 256 \
                --dump "../final/cu$cu-$round-$phase.f32" \
                > "../final/cu$cu-$round-$phase.jsonl" 2> "../final/cu$cu-$round-$phase.log"
        done
        python3 tools/roofline-compare.py "../final/cu$cu-$round-a.f32" "../final/cu$cu-$round-b.f32" \
            > "../final/cu$cu-$round-parity.json"
        cmp "../final/cu$cu-$round-a.f32" "../final/cu$cu-$round-a2.f32"
        cmp "../final/cu$cu-$round-a.f32.gdn" "../final/cu$cu-$round-a2.f32.gdn"
        rm ../final/cu"$cu"-"$round"-*.f32 ../final/cu"$cu"-"$round"-*.f32.gdn
    done
    for round in 1 2 3 4 5; do
        for mask in 0 27; do
            FG_ROOFLINE=$mask FG_ROOF_DENSE=7 FG_ROOF_QSA=3 ./tools/bench-decode-fusion \
                --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
                > "../final/cu$cu-profile-$round-$mask.jsonl" 2> "../final/cu$cu-profile-$round-$mask.log"
        done
    done
    for mask in 0 27; do
        FG_ROOFLINE=$mask FG_ROOF_DENSE=7 FG_ROOF_QSA=3 ./tools/bench-decode-fusion \
            --q4 --profile --warmup 2048 --iterations 1 --windows 1 --seed 7 \
            > "../final/cu$cu-long-$mask.jsonl" 2> "../final/cu$cu-long-$mask.log"
        FG_ROOFLINE=$mask FG_ROOF_DENSE=7 FG_ROOF_QSA=3 ./tools/bench-decode-fusion \
            --q4 --no-clocks --warmup 256 --iterations 64 --windows 3 --seed 7 \
            > "../final/cu$cu-no-clocks-$mask.jsonl" 2> "../final/cu$cu-no-clocks-$mask.log"
    done
    for mask in 0 27; do
        for filter in dense_quant_compact decode_fusion_router roof_qsa roof_controls hc_inject_partial \
            gr_partial_boundaries gdn_algebraic expert_decode_fused_random q8_decode_shape_parity \
            batch_submission_parity pipeline_flush_parity static_batch_replay; do
            FG_ROOFLINE=$mask FG_ROOF_QSA=3 DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk \
                > "../final/cu$cu-test-$mask-$filter.txt" 2>&1
        done
    done
    cat /sys/class/drm/card*/device/pp_dpm_sclk > "../final/cu$cu-sclk-end.txt"
done
sha256sum tools/bench-decode-fusion flash-gordon > ../final/executable-sha256.txt
cd ..
restore
trap - EXIT
tar -cf final.tar -C final .
cat final/restored.txt final/health-8091.json final/health-8092.json
