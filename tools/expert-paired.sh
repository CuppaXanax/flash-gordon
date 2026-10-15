#!/bin/sh
set -eu
case " $(hostname -I) " in *" 192.168.42.50 "*) ;; *) exit 64;; esac
cd /home/xander/hc-experts
export XDG_CACHE_HOME="$PWD/cache"
tar -xf candidate.tar
restore() {
    test ! -f candidate-vk.c || cp candidate-vk.c src/vk.c
    test ! -f candidate-test.c || cp candidate-test.c tests/test_fg_vk.c
}
trap restore EXIT
python3 tools/expert-measure.py
touch shaders/fg_moe_decode_gate_up.comp shaders/fg_moe_decode_down_reduce.comp shaders/fg_moe_decode_down_q8.comp
make -s -j8 tests/test_fg_vk > build-paired.log 2>&1 || { tail -60 build-paired.log; exit 1; }
label="paired-$(date +%s)"
status=0
for format in q5 q4q5 q4q8; do
    FG_BENCH_EXPERT_FORMAT="$format" FG_BENCH_EXPERT_DECODE=30 FG_EXPERT_PARITY_DUMP="$label.bin" FG_EXPERT_BOUND_DUMP="$label.bounds" DS4_REMOTE_TEST_FILTER=expert_decode_pair_benchmark ./tests/test_fg_vk >> "$label.log" 2>&1 || { status=$?; break; }
done
cat "$label.log"
python3 - "$label.bin" "$label.bounds" <<'PY'
import array, math, sys
a = array.array("f")
with open(sys.argv[1], "rb") as f:
    a.frombytes(f.read())
assert len(a) == 3 * 30 * 2 * 2560
b = array.array("d")
with open(sys.argv[2], "rb") as f:
    b.frombytes(f.read())
assert len(b) == len(a)
worst = 0.0
for base in range(0, len(a), 5120):
    for j, (x, y) in enumerate(zip(a[base:base+2560], a[base+2560:base+5120])):
        assert math.isfinite(x) and math.isfinite(y)
        if 60*2560 <= base < 120*2560:
            assert x == y, (base, j, x, y)
        rel = abs(x-y)/max(1.0,b[base+j],b[base+2560+j])
        worst = max(worst, rel)
        assert rel <= 4e-5, (base, x, y, rel)
print("BASELINE_PARITY max_conditioned=", worst, "PASS")
PY
exit "$status"
