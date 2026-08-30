#!/usr/bin/env bash
set -euo pipefail

root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
token=30
coordinator="$root/coordinator.log"
touch "$coordinator"
for rank in {1..7}; do touch "$root/rank-$rank.log"; done

layer_submissions=0
layer_dispatches=0
for layer in {0..47}; do
    rank=$(((layer + 1) % 8))
    local=0
    remotes=1
    local_selected=0
    submissions=2
    dispatches=24
    if ((rank == 0)); then
        local=1
        remotes=0
        local_selected=10
        submissions=3
        dispatches=28
    fi
    if ((layer == 1)); then
        submissions=$((submissions + 1))
        dispatches=$((dispatches + 8))
    fi
    if ((layer == 47)); then
        submissions=$((submissions + 1))
    fi
    mask=$((1 << rank))
    printf 'EP_ROUTE_TRACE token=%u layer=%u routes=1 remotes=%u local=%u local_selected=%u selected=10 rank_mask=%u\n' "$token" "$layer" "$remotes" "$local" "$local_selected" "$mask" >> "$coordinator"
    printf 'EP_LAYER_TRACE token=%u layer=%u status=0 total_ms=5.000 sync1_ms=1.000 fire_ms=1.000 shared_ms=1.000 collect_ms=1.000 finish_ms=1.000 submissions=%u dispatches=%u\n' "$token" "$layer" "$submissions" "$dispatches" >> "$coordinator"
    if ((rank != 0)); then
        printf 'WORKER_EXPERT layer[%u] t=%u rank=%u sel=10 decode=0.01 gpu=0.20 reduce=0.01 encode=0.01 send=0.01 total=0.24\n' "$layer" "$token" "$rank" >> "$root/rank-$rank.log"
    fi
    layer_submissions=$((layer_submissions + submissions))
    layer_dispatches=$((layer_dispatches + dispatches))
done
printf 'TOKEN_PROFILE rank=0 token=%u kind=token layer=4294967295 wall_ms=250.000 gpu_ms=100.000 kernel_ms=98.000 vk_overhead_ms=2.000 wall_residual_ms=150.000 submissions=%u dispatches=%u\n' "$token" "$((layer_submissions + 2))" "$((layer_dispatches + 2))" >> "$coordinator"

awk -v expected_token="$token" -f tests/validate_ep_trace.awk "$coordinator" "$root"/rank-*.log >/dev/null
sed -i '0,/selected=10/s//selected=9/' "$coordinator"
if awk -v expected_token="$token" -f tests/validate_ep_trace.awk "$coordinator" "$root"/rank-*.log >/dev/null 2>&1; then
    echo "corrupt EP trace unexpectedly passed" >&2
    exit 1
fi
echo "Flash Gordon EP trace validator: PASS"