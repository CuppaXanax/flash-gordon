#!/bin/bash
set -euo pipefail
cd /home/xander/hc-roof
mkdir -p src evidence incremental cumulative .build cache
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf source.tar -C src
cd src
mkdir -p .build
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk tests/test_q4_repack > ../incremental/build.txt 2>&1
bash tools/check-flags.sh > ../incremental/flags.txt
./tests/test_q4_repack > ../incremental/repack.txt
cd ..
candidate=2047 candidate_expert=33024 candidate_hc=96 candidate_recurrent=3 candidate_gdn=1
gp= restricted=0
restore(){
    cd /home/xander/hc-roof
    local status=0
    if [ "$restricted" = 1 ]; then
        /usr/local/bin/bc250-cu-live-manager --yes apply-service > incremental/cu-restored-apply.txt 2>&1 || status=1
        restricted=0
    fi
    /usr/local/bin/bc250-cu-live-manager status > incremental/cu-restored-status.txt
    grep -q 'CUs active & routed  : 40/40' incremental/cu-restored-status.txt || status=1
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; gp=; fi
    systemctl start cyan-skillfish-governor-smu bc250-embed bc250-rerank || status=1
    cmp evidence/original.toml /etc/cyan-skillfish-governor-smu/config.toml || status=1
    systemctl is-active cyan-skillfish-governor-smu bc250-embed bc250-rerank > incremental/restored.txt || status=1
    for port in 8091 8092; do
        ready=0
        for attempt in $(seq 1 90); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" > "incremental/health-$port.json"; then ready=1; break; fi
            sleep 2
        done
        [ "$ready" = 1 ] || status=1
    done
    return "$status"
}
finish(){
    rc=$?
    restore || rc=1
    tar -czf excess-final.tar.gz --exclude='*.f32' --exclude='*.gdn' --exclude=governor.log \
        evidence excess excess2 excess3 excess4 excess5 incremental cumulative
    exit "$rc"
}
trap finish EXIT
/usr/local/bin/bc250-cu-live-manager status > incremental/cu-original-status.txt
grep -q 'CUs active & routed  : 40/40' incremental/cu-original-status.txt
systemctl stop bc250-embed bc250-rerank cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/evidence/pinned.toml" > incremental/governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cd src
run(){
    local mask=$1; shift
    local expert=2 hc=0 recurrent=0 gdn=8
    if [ "$mask" = "$candidate" ]; then expert=$candidate_expert hc=$candidate_hc recurrent=$candidate_recurrent gdn=$candidate_gdn; fi
    FG_ROOFLINE=$mask FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=$expert \
        FG_ROOF_HC=$hc FG_ROOF_RECURRENT=$recurrent FG_ROOF_GDN=$gdn "$@"
}
for cu in 40 24; do
    if [ "$cu" = 24 ]; then
        restricted=1
        /usr/local/bin/bc250-cu-live-manager --yes stock-dispatch > ../incremental/cu24-apply.txt
        /usr/local/bin/bc250-cu-live-manager status > ../incremental/cu24-status.txt
        grep -q 'CUs active & routed  : 24/40' ../incremental/cu24-status.txt
    fi
    for study in incremental cumulative; do
        baseline=251
        if [ "$study" = cumulative ]; then baseline=0; fi
        for round in 1 2 3 4 5; do
            for phase in a b a2; do
                mask=$baseline
                if [ "$phase" = b ]; then mask=$candidate; fi
                run "$mask" ./tools/bench-decode-fusion --q4 --seed "$round" \
                    --windows 5 --iterations 64 --warmup 256 --dump "../$study/cu$cu-$round-$phase.f32" \
                    > "../$study/cu$cu-$round-$phase.jsonl" 2> "../$study/cu$cu-$round-$phase.log"
            done
            python3 tools/roofline-compare.py "../$study/cu$cu-$round-a.f32" "../$study/cu$cu-$round-b.f32" \
                > "../$study/cu$cu-$round-parity.json"
            cmp "../$study/cu$cu-$round-a.f32" "../$study/cu$cu-$round-a2.f32"
            cmp "../$study/cu$cu-$round-a.f32.gdn" "../$study/cu$cu-$round-a2.f32.gdn"
            rm ../"$study"/cu"$cu"-"$round"-*.f32 ../"$study"/cu"$cu"-"$round"-*.f32.gdn
        done
        for round in 1 2 3 4 5; do
            for mask in "$baseline" "$candidate"; do
                run "$mask" ./tools/bench-decode-fusion --q4 --profile --warmup 256 --iterations 1 --windows 1 --seed "$round" \
                    > "../$study/cu$cu-profile-$round-$mask.jsonl" 2> "../$study/cu$cu-profile-$round-$mask.log"
            done
        done
        for mask in "$baseline" "$candidate"; do
            for label in no-clocks long; do
                extra=(--no-clocks --warmup 256 --iterations 64 --windows 3)
                if [ "$label" = long ]; then extra=(--profile --warmup 2048 --iterations 1 --windows 1); fi
                run "$mask" ./tools/bench-decode-fusion --q4 --seed 7 "${extra[@]}" \
                    > "../$study/cu$cu-$label-$mask.jsonl" 2> "../$study/cu$cu-$label-$mask.log"
            done
        done
    done
    for mask in 251 "$candidate"; do
        for filter in roof_hc_up_mix roof_hc_down_inject roof_gdn_gate_controls roof_dense_exact \
            dense_quant_compact gdn_algebraic expert_decode_fused_random roof_qsa roof_controls \
            hc_inject_partial gr_partial_boundaries q8_decode_shape_parity batch_submission_parity \
            pipeline_flush_parity static_batch_replay; do
            DS4_REMOTE_TEST_FILTER=$filter run "$mask" ./tests/test_fg_vk \
                > "../incremental/cu$cu-test-$mask-$filter.txt" 2>&1
        done
        FG_BENCH_HC_INJECT=1 DS4_REMOTE_TEST_FILTER=roof_hc_up_mix run "$candidate" ./tests/test_fg_vk \
            > "../incremental/cu$cu-hc-up-shape-bench.txt" 2>&1
        FG_ROOFLINE=0 FG_ROOF_HC=2 DS4_REMOTE_TEST_FILTER=q8_decode_shape_parity ./tests/test_fg_vk \
            > "../incremental/cu$cu-rejected-q8-rows-oracle.txt" 2>&1
        FG_BENCH_EXPERT_DECODE=80 FG_BENCH_EXPERT_FORMAT=q4q5 DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark \
            run "$mask" ./tests/test_fg_vk > "../incremental/cu$cu-expert-$mask.txt" 2>&1
    done
    cat /sys/class/drm/card*/device/pp_dpm_sclk > "../incremental/cu$cu-sclk-end.txt"
done
