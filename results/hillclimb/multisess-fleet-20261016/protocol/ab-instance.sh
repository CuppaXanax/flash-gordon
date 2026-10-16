#!/usr/bin/env bash
set -euo pipefail
action=${1:?action}
release=${2:?release}
manifest=${3:?manifest}
sessions=${4:-1}
wide=${5:-0}
[[ $release == /home/xander/flash-gordon-live/20260926-abl2 ||
   $release =~ ^/home/xander/flash-gordon-live/[0-9]{8}-abq[-a-zA-Z0-9]+$ ]]
[[ $manifest == /home/xander/fg-ring-pack-abliterated/manifest.fgm ||
   $manifest == /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm ]]
[[ $sessions == 1 || $sessions == 2 ]]
[[ $wide == 0 || $wide == 1 ]]
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
((rank >= 0 && rank < 8))
run=/home/xander/flash-gordon-live/20261015-abq-operator
case "$action" in
status)
    sha256sum "$release/flash-gordon" "$manifest"
    for proc in /proc/[0-9]*; do
        test -O "$proc" || continue
        exe=$(readlink "$proc/exe" 2>/dev/null) || continue
        [[ $exe == /home/xander/flash-gordon-live/*/flash-gordon ]] || continue
        printf 'PID=%s EXE=%s CMD=' "${proc##*/}" "$exe"
        tr '\0' ' ' <"$proc/cmdline"; echo
    done
    grep -E 'MemAvailable|SwapFree|SwapTotal' /proc/meminfo
    for device in /sys/class/drm/card[0-9]*/device; do
        for value in "$device"/pp_dpm_sclk "$device"/pp_dpm_mclk "$device"/gpu_busy_percent "$device"/hwmon/hwmon*/temp1_input; do
            if [[ -r $value ]]; then echo "SENSOR $value"; cat "$value" || echo 'SENSOR unavailable'; fi
        done
    done
    df -B1 /home/xander
    ;;
stop)
    for proc in /proc/[0-9]*; do
        test -O "$proc" || continue
        exe=$(readlink "$proc/exe" 2>/dev/null) || continue
        [[ $exe == /home/xander/flash-gordon-live/*/flash-gordon ]] || continue
        mapfile -d '' -t args <"$proc/cmdline" || continue
        for ((i=0; i+1<${#args[@]}; i++)); do
            if [[ ${args[i]} == --manifest && ${args[i+1]} == "$manifest" ]]; then
                pid=${proc##*/}
                [[ $exe == "$release/flash-gordon" ]] || { echo "Unexpected live release $exe" >&2; exit 1; }
                kill -TERM "$pid"
                for ((waited=0; waited<45; waited++)); do
                    [[ -e $proc/exe ]] || break
                    [[ $(awk '{print $3}' "$proc/stat") == Z ]] && break
                    sleep 1
                done
                if [[ -e $proc/exe && $(awk '{print $3}' "$proc/stat") != Z ]]; then
                    echo "PID $pid did not stop; no forced kill" >&2; exit 1
                fi
                echo "STOPPED rank=$rank pid=$pid"
                break
            fi
        done
    done
    ;;
start)
    for proc in /proc/[0-9]*; do
        test -O "$proc" || continue
        exe=$(readlink "$proc/exe" 2>/dev/null) || continue
        if [[ $exe == /home/xander/flash-gordon-live/*/flash-gordon ]]; then
            echo "Serving PID ${proc##*/} already exists; stop its exact manifest first" >&2
            exit 1
        fi
    done
    mkdir -p "$run"
    if [[ -e $run/rank-$rank.log ]]; then
        [[ -f $run/rank-$rank.log && ! -L $run/rank-$rank.log ]]
        mv "$run/rank-$rank.log" "$run/previous-rank-$rank.log"
        if [[ -f $run/rank-$rank.pid ]]; then
            cp "$run/rank-$rank.pid" "$run/previous-rank-$rank.pid"
        fi
    fi
    cd "$release"
    args=(rank --manifest "$manifest" --rank "$rank")
    if ((rank == 0)); then
        args=(api --manifest "$manifest" --model-name Qwen3.8-Flash-Next-Uncensored
              --host 0.0.0.0 --port 8080 --context-tokens 262144 --gpu-index-tokens 262144
              --qsa-hot-tokens 0 --qsa-page-cache-mib 700)
        if ((sessions == 2)); then args+=(--sessions 2); fi
    fi
    flags=(FG_SHADER_DIR="$release/vulkan" FG_EXPERT_Q8_WIDE="$wide")
    if ((rank == 0)); then flags+=(FG_RING_TRACE=1); fi
    # AB_STACK_FLAGS
    # AB_QUALITY_CAPTURE
    nohup env -u FG_ROOFLINE -u FG_ROOF_DENSE -u FG_ROOF_QSA -u FG_ROOF_EXPERT \
        -u FG_ROOF_HC -u FG_ROOF_RECURRENT -u FG_ROOF_GDN \
        -u FG_NGRAM_LOCAL_CACHE -u FG_EVAL_LOGITS -u FG_EVAL_LOGITS_DIR \
        -u FG_DECODE_PROFILE -u FG_FRAME_TRACE -u FG_DECODE_RING_TRACE \
        -u FG_FABRIC_PROFILE -u FG_GENERATE_TRACE -u FG_OUTPUT_SPLIT_TRACE \
        -u FG_DECODE_FUSION -u FG_DECODE_SUBMIT_GROUP -u FG_RING_TRACE -u FG_DENSE_SOA -u FG_DENSE_K_COOKED -u LD_PRELOAD \
        "${flags[@]}" "$release/flash-gordon" "${args[@]}" >"$run/rank-$rank.log" 2>&1 </dev/null &
    pid=$!
    echo "$pid" >"$run/rank-$rank.pid"
    sleep 2
    kill -0 "$pid"
    echo "STARTED rank=$rank pid=$pid"
    ;;
*) exit 2 ;;
esac
