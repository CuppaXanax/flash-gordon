#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
mkdir -p .build/stack/cache
export TMPDIR="$PWD/.build/stack"
export MESA_SHADER_CACHE_DIR="$PWD/.build/stack/cache"
export XDG_CACHE_HOME="$PWD/.build/stack/cache"
make -s -j6 all tools/q4-eval-batch tests/test_fg_vk tests/test_sampler tests/test_chat tests/test_chat_runtime tests/test_qsa_prefill
bash tools/check-flags.sh
for test in test_sampler test_chat test_chat_runtime; do
    "./tests/$test"
    printf 'CPU %s: PASS\n' "$test"
done
tests=("$@")
if (($# == 0)); then
    tests=(roof_controls expert_decode_fused_q8_gates ngram_direct_lookup submit_group)
fi
for test in "${tests[@]}"; do
    if DS4_REMOTE_TEST_FILTER="$test" ./tests/test_fg_vk; then
        :
    else
        status=$?
        if [[ $status == 77 ]]; then
            printf '%s\n' 'Local Vulkan unavailable; hardware oracle execution remains pending.'
            break
        fi
        exit "$status"
    fi
done
if ./tests/test_qsa_prefill; then
    :
else
    status=$?
    if [[ $status != 77 ]]; then exit "$status"; fi
    printf '%s\n' 'Local Vulkan unavailable; QSA readiness/upload oracles remain pending.'
fi
sha256sum flash-gordon tools/q4-eval-batch
