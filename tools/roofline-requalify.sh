#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p requalified readmitted
export XDG_CACHE_HOME=$PWD/cache TMPDIR=$PWD/.build
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk > ../requalified/build.txt 2>&1
bash tools/check-flags.sh > ../requalified/flags.txt
cd ..
gp= restricted=0
restore(){
    cd /home/xander/hc-roof
    local status=0
    if [ "$restricted" = 1 ]; then
        /usr/local/bin/bc250-cu-live-manager --yes apply-service > requalified/cu-restored-apply.txt 2>&1 || status=1
        restricted=0
    fi
    /usr/local/bin/bc250-cu-live-manager status > requalified/cu-restored-status.txt
    grep -q 'CUs active & routed  : 40/40' requalified/cu-restored-status.txt || status=1
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank || status=1
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml || status=1
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > requalified/restored.txt || status=1
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "requalified/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || status=1
    done
    return "$status"
}
trap restore EXIT
/usr/local/bin/bc250-cu-live-manager status > requalified/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' requalified/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > requalified/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
for cu in 40 24; do
    if [ "$cu" = 24 ]; then
        restricted=1
        /usr/local/bin/bc250-cu-live-manager --yes stock-dispatch > ../requalified/cu24-apply.txt
        /usr/local/bin/bc250-cu-live-manager status > ../requalified/cu24-status.txt
        grep -q 'CUs active & routed  : 24/40' ../requalified/cu24-status.txt
    fi
    for study in readmitted requalified; do
        candidate=219 dense=6 expert=0
        if [ "$study" = requalified ]; then candidate=251; dense=19; expert=2; fi
        for round in 1 2 3 4 5; do
            for phase in a b a2; do
                mask=0
                if [ "$phase" = b ]; then mask=$candidate; fi
                FG_ROOFLINE=$mask FG_ROOF_DENSE=$dense FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert \
                    ./tools/bench-decode-fusion --q4 --seed "$round" --windows 5 --iterations 64 --warmup 256 \
                    --dump "../$study/cu$cu-$round-$phase.f32" \
                    > "../$study/cu$cu-$round-$phase.jsonl" 2> "../$study/cu$cu-$round-$phase.log"
            done
            python3 tools/roofline-compare.py "../$study/cu$cu-$round-a.f32" "../$study/cu$cu-$round-b.f32" \
                > "../$study/cu$cu-$round-parity.json"
            cmp "../$study/cu$cu-$round-a.f32" "../$study/cu$cu-$round-a2.f32"
            cmp "../$study/cu$cu-$round-a.f32.gdn" "../$study/cu$cu-$round-a2.f32.gdn"
            rm ../"$study"/cu"$cu"-"$round"-*.f32 ../"$study"/cu"$cu"-"$round"-*.f32.gdn
        done
        for round in 1 2 3 4 5; do
            for mask in 0 "$candidate"; do
                FG_ROOFLINE=$mask FG_ROOF_DENSE=$dense FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert \
                    ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
                    > "../$study/cu$cu-profile-$round-$mask.jsonl" 2> "../$study/cu$cu-profile-$round-$mask.log"
            done
        done
        for mask in 0 "$candidate"; do
            for label in no-clocks long; do
                extra=(--no-clocks --warmup 256 --iterations 64 --windows 3)
                if [ "$label" = long ]; then extra=(--profile --warmup 2048 --iterations 1 --windows 1); fi
                FG_ROOFLINE=$mask FG_ROOF_DENSE=$dense FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert \
                    ./tools/bench-decode-fusion --q4 --seed 7 "${extra[@]}" \
                    > "../$study/cu$cu-$label-$mask.jsonl" 2> "../$study/cu$cu-$label-$mask.log"
            done
            for filter in dense_quant_compact decode_fusion_router roof_qsa roof_controls hc_inject_partial \
                gr_partial_boundaries gdn_algebraic expert_decode_fused_random q8_decode_shape_parity \
                batch_submission_parity pipeline_flush_parity static_batch_replay; do
                FG_ROOFLINE=$mask FG_ROOF_DENSE=$dense FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert \
                    DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk > "../$study/cu$cu-test-$mask-$filter.txt" 2>&1
            done
        done
    done
    cat /sys/class/drm/card*/device/pp_dpm_sclk > "../requalified/cu$cu-sclk-end.txt"
done
cd ..
restore
trap - EXIT
tar -czf followup.tar.gz --exclude='*.f32' --exclude='*.gdn' --exclude=governor.log \
    evidence bandwidth readmitted requalified
cat requalified/restored.txt requalified/health-8091.json requalified/health-8092.json
