#!/usr/bin/env bash
# Adapted from cuppaxanax/decode-climb's manifest-scoped instance helper.
set -euo pipefail
action=${1:?action}
release=${2:?release}
if [[ $action == replay ]]; then
  [[ $release =~ ^/home/xander/flash-gordon-live/[0-9]{8}-host-overhead[-a-zA-Z0-9]*$ ]]
  blade=$(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)
  [[ $blade =~ ^5[0-3]$ ]]
  cd "$release"
  mkdir -p .build-scratch
  export TMPDIR="$PWD/.build-scratch"
  trap 'rmdir "$TMPDIR"' EXIT
  DS4_REMOTE_TEST_FILTER=submit_group ./tests/test_fg_vk
  for filter in batch_submission_parity pipeline_flush_parity static_batch_replay gpu_profile; do
    DS4_REMOTE_TEST_FILTER="$filter" ./tests/test_fg_vk
  done
  ./tests/test_qsa_prefill
  DS4_REMOTE_TEST_FILTER=submit_group_benchmark ./tests/test_fg_vk
  exit
fi
mode=${3:-plain}
manifest=${4:-/home/xander/fg-ring-pack-abliterated/manifest.fgm}
tag=${5:-run}
[[ $release =~ ^/home/xander/flash-gordon-live/[0-9]{8}-q4[-a-zA-Z0-9]*$ ||
   $release == /home/xander/flash-gordon-live/20260926-abl2 ]]
[[ $manifest =~ ^/home/xander/fg-ring-pack-abliterated(-q4(-protected|-teacher|-control|-ablate-[a-z0-9-]+|-mixed)?)?/manifest\.fgm$ ]]
[[ $tag =~ ^[a-z0-9-]+$ ]]
case "$mode" in plain|trace|profile|prefill|quality|numerics) ;; *) exit 2 ;; esac
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
((rank >= 0 && rank < 8))
run="$release/operator"
if [[ $release == /home/xander/flash-gordon-live/20260926-abl2 ]]; then
  run="/home/xander/flash-gordon-live/$(date +%Y%m%d)-q4-original-operator"
fi
log="$run/rank-$rank-$mode-$tag.log"
case "$action" in
stop)
  for proc in /proc/[0-9]*; do
    test -O "$proc" || continue
    exe=$(readlink "$proc/exe" 2>/dev/null) || continue
    case "$exe" in /home/xander/flash-gordon-live/*/flash-gordon|/home/xander/flash-gordon-live/*/q4-eval-batch) ;; *) continue ;; esac
    mapfile -d '' -t args < "$proc/cmdline" || continue
    for ((i=0; i+1<${#args[@]}; i++)); do
      if [[ ${args[i]} == --manifest && ${args[i+1]} == "$manifest" ]]; then
        pid=${proc##*/}
        echo "Stopping rank=$rank pid=$pid exe=$exe manifest=$manifest"
        kill -TERM "$pid"
        for ((waited=0; waited<30; waited++)); do
          [[ -e "$proc/exe" ]] || break
          [[ $(awk '{print $3}' "$proc/stat") == Z ]] && break
          sleep 1
        done
        if [[ -e "$proc/exe" && $(awk '{print $3}' "$proc/stat") != Z ]]; then
          echo "Process $pid did not stop; refusing forced kill" >&2; exit 1
        fi
        break
      fi
    done
  done
  ;;
start)
  mkdir -p "$run"
  cd "$release"
  args=(rank --manifest "$manifest" --rank "$rank")
  if ((rank == 0)); then
    args=(api --manifest "$manifest" --model-name Qwen3.8-Flash-Next-Uncensored
      --host 0.0.0.0 --port 8080 --context-tokens 262144 --gpu-index-tokens 262144
      --qsa-hot-tokens 0 --qsa-page-cache-mib 700)
  fi
  flags=(FG_SHADER_DIR="$release/vulkan" MESA_SHADER_CACHE_DIR="$run/.mesa-cache")
  if [[ $mode == profile ]]; then
    flags+=(FG_DECODE_PROFILE=1 FG_DECODE_RING_TRACE=1 FG_FABRIC_PROFILE=1 FG_GENERATE_TRACE=1)
  elif [[ $mode == prefill ]]; then
    flags+=(FG_PREFILL_PROFILE=1)
  elif [[ $mode == trace ]]; then
    flags+=(FG_DECODE_RING_TRACE=1 FG_FABRIC_PROFILE=1 FG_GENERATE_TRACE=1)
  elif [[ $mode == numerics ]]; then
    flags+=(FG_NUMERICS_TRACE=1 FG_GDN_DIAG=1)
  elif [[ $mode == quality ]] && ((rank == 4)); then
    mkdir "$run/logits-$tag"
    flags+=(FG_EVAL_LOGITS_DIR="$run/logits-$tag")
  fi
  if ((rank == 0)); then flags+=(FG_RING_TRACE=1); fi
  nohup env -u FG_DENSE_SOA -u FG_DENSE_K_COOKED -u FG_EVAL_LOGITS \
    -u FG_EVAL_LOGITS_DIR -u FG_DECODE_PROFILE -u FG_DECODE_RING_TRACE \
    -u FG_FABRIC_PROFILE -u FG_GENERATE_TRACE -u FG_NUMERICS_TRACE -u FG_GDN_DIAG \
    -u FG_PREFILL_PROFILE -u FG_PREFILL_PROFILE_CHUNK -u FG_PROFILE_TOKEN \
    "${flags[@]}" "$release/flash-gordon" "${args[@]}" >"$log" 2>&1 </dev/null &
  pid=$!
  echo "$pid" >"$run/rank-$rank.pid"
  sleep 2
  kill -0 "$pid"
  echo "Started rank=$rank pid=$pid release=$release manifest=$manifest mode=$mode tag=$tag"
  ;;
logs) cat "$log" ;;
status)
  cat "$run/rank-$rank.pid"
  tail -5 "$log"
  ;;
*) exit 2 ;;
esac
