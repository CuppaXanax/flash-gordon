#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
cd /home/xander/hc-experts
export XDG_CACHE_HOME="$PWD/cache"
tar -xf candidate.tar
mkdir -p fixtures
cp tests/test_core.c core-original.c
trap 'cp core-original.c tests/test_core.c' EXIT
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_core.c")
p.write_bytes(p.read_bytes().replace(b"/tmp/", b"/home/xander/hc-experts/fixtures/"))
PY
touch src/vk.c tests/test_fg_vk.c shaders/fg_moe_decode_gate_up.comp shaders/fg_moe_decode_down_reduce.comp shaders/fg_moe_decode_down_q8.comp
make -s -j8 flash-gordon tests/test_fg_vk tests/test_core > build-validate.log 2>&1 || { tail -60 build-validate.log; exit 1; }
./tests/test_core > core.log 2>&1 || { cat core.log; exit 1; }
tail -5 core.log
for filter in expert_decode_down_exact expert_decode_fused expert_decode_fused_q8_0 expert_decode_fused_q8_0_cooked expert_decode_fused_q8_gates expert_decode_fused_random expert_decode_fused_wide expert_graph_fused expert_graph_replay kquant_cooked q5_1_down_cooked decode_tile_schedule static_batch_replay batch_submission_parity pipeline_flush_parity; do
    DS4_REMOTE_TEST_FILTER="$filter" ./tests/test_fg_vk > "validate-$filter.log" 2>&1 || { cat "validate-$filter.log"; exit 1; }
    cat "validate-$filter.log"
done
systemctl is-active bc250-embed bc250-rerank
