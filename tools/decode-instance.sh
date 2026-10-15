#!/usr/bin/env bash
set -euo pipefail

action=${1:?action}
release=${2:?release}
mode=${3:-plain}
if [[ ! $release =~ ^/home/xander/flash-gordon-live/[0-9]{8}-decode[-a-zA-Z0-9]*$ &&
      $release != /home/xander/flash-gordon-live/20260926-abl2 ]]; then exit 2; fi
case "$mode" in plain|trace|profile|control|select) ;; *) exit 2 ;; esac
pack=/home/xander/fg-ring-pack-abliterated
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
((rank >= 0 && rank < 8))
run="$release/operator"
if [[ $release == /home/xander/flash-gordon-live/20260926-abl2 ]]; then
    run="/home/xander/flash-gordon-live/$(date +%Y%m%d)-decode-operator"
fi

case "$action" in
stage)
    [[ $release != /home/xander/flash-gordon-live/20260926-abl2 && ! -e $release/flash-gordon ]]
    mkdir -p "$release"
    cp /home/xander/flash-gordon-live/20260926-abl2/flash-gordon "$release/"
    cp -a /home/xander/flash-gordon-live/20260926-abl2/vulkan "$release/"
    sha256sum "$release/flash-gordon" "$pack/manifest.fgm"
    ;;
stop)
    # Match the operator's manifest-scoped stop path, not a process name.
    for proc in /proc/[0-9]*; do
        test -O "$proc" || continue
        exe=$(readlink "$proc/exe" 2>/dev/null) || continue
        case "$exe" in /home/xander/flash-gordon-live/*/flash-gordon) ;; *) continue ;; esac
        mapfile -d '' -t args < "$proc/cmdline" || continue
        for ((i=0; i+1<${#args[@]}; i++)); do
            if [[ ${args[i]} == --manifest && ${args[i+1]} == "$pack/manifest.fgm" ]]; then
                pid=${proc##*/}
                echo "Stopping rank=$rank pid=$pid exe=$exe"
                kill -TERM "$pid"
                for ((waited=0; waited<30; waited++)); do
                    [[ -e "$proc/exe" ]] || break
                    [[ $(awk '{print $3}' "$proc/stat") == Z ]] && break
                    sleep 1
                done
                if [[ -e "$proc/exe" && $(awk '{print $3}' "$proc/stat") != Z ]]; then
                    echo "Process $pid did not stop; refusing forced kill" >&2
                    exit 1
                fi
                break
            fi
        done
    done
    ;;
start)
    mkdir -p "$run"
    cd "$release"
    args=(rank --manifest "$pack/manifest.fgm" --rank "$rank")
    if ((rank == 0)); then
        args=(api --manifest "$pack/manifest.fgm" --model-name Qwen3.8-Flash-Next-Uncensored
            --host 0.0.0.0 --port 8080 --context-tokens 262144 --gpu-index-tokens 262144
            --qsa-hot-tokens 0 --qsa-page-cache-mib 700)
    fi
    flags=(FG_SHADER_DIR="$release/vulkan")
    if [[ $mode == profile ]]; then
        flags+=(FG_DECODE_PROFILE=1 FG_DECODE_RING_TRACE=1 FG_FABRIC_PROFILE=1 FG_GENERATE_TRACE=1)
    elif [[ $mode == trace || $mode == control || $mode == select ]]; then
        flags+=(FG_DECODE_RING_TRACE=1 FG_FABRIC_PROFILE=1 FG_GENERATE_TRACE=1)
    fi
    if [[ $mode == control ]]; then flags+=(FG_QSA_DEVICE_SELECT=0); fi
    if ((rank == 0)); then flags+=(FG_RING_TRACE=1); fi
    nohup env "${flags[@]}" "$release/flash-gordon" "${args[@]}" >"$run/rank-$rank-$mode.log" 2>&1 </dev/null &
    pid=$!
    echo "$pid" >"$run/rank-$rank.pid"
    sleep 2
    kill -0 "$pid"
    echo "Started rank=$rank pid=$pid release=$release mode=$mode"
    ;;
logs)
    [[ -f $run/rank-$rank-$mode.log ]] || run=/home/xander/hc-decode
    cat "$run/rank-$rank-$mode.log"
    ;;
status)
    [[ -f $run/rank-$rank.pid ]] || run=/home/xander/hc-decode
    cat "$run/rank-$rank.pid"
    tail -5 "$run/rank-$rank-$mode.log"
    ;;
*) exit 2 ;;
esac
