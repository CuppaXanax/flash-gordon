#!/usr/bin/env bash
# Flag-budget gate for docs/FLAGS.md.
#
# Enumerates every getenv("FG_*") read in src/ and include/ (the serving and
# packer code) plus tests/, and fails when:
#   * a code-read flag has no live row in docs/FLAGS.md,
#   * a live row names a flag no code reads,
#   * the code set differs from the checked-in expected set below (so a new
#     flag cannot merge without an explicit update here and a doc row),
#   * a live row has a status outside the policy set in the doc header,
#   * a flag documented under "## Removed flags" is still read by code.
#
# The expected sets are intentionally explicit: a new environment flag must
# touch this file and docs/FLAGS.md in the same change.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
doc=docs/FLAGS.md
failures=0

fail() {
    printf 'FGFLAGS failure: %s\n' "$*" >&2
    failures=$((failures + 1))
}

extract() {
    { grep -rhoE 'getenv\("FG_[A-Z0-9_]+"' "$@" || true; } |
        tr -d '\r' | sed -E 's/getenv\("(FG_[A-Z0-9_]+)"/\1/' | LC_ALL=C sort -u
}

expected_code="$(printf '%s\n' \
    FG_BLOCK_BENCH \
    FG_DECODE_MS \
    FG_DECODE_PROFILE \
    FG_DECODE_RING \
    FG_DECODE_RING_TRACE \
    FG_DECODE_STATIC \
    FG_FABRIC_PROFILE \
    FG_FRAME_TRACE \
    FG_GDN_DIAG \
    FG_GENERATE_TRACE \
    FG_NGRAM_LOCALITY_TRACE \
    FG_NGRAM_VERIFY \
    FG_NUMERICS_TRACE \
    FG_OUTPUT_SPLIT \
    FG_OUTPUT_SPLIT_TIMEOUT_MS \
    FG_OUTPUT_SPLIT_TRACE \
    FG_OUTPUT_TRACE \
    FG_PACK_EMBED_RANK \
    FG_PREFILL_PROFILE \
    FG_PREFILL_PROFILE_CHUNK \
    FG_PREFILL_RING \
    FG_PREFIX_TRACE \
    FG_PROFILE_TOKEN \
    FG_QSA_LOCALITY_MIB \
    FG_QSA_LOCALITY_TRACE \
    FG_QSA_PREFETCH_TRACE \
    FG_QSA_WARM_TRACE \
    FG_RING_TRACE \
    FG_SET_TRACE \
    FG_SHADER_DIR \
    FG_STATIC_CHECK \
    FG_STATIC_RERECORD \
    FG_TRACE_ROUTES \
    FG_WORKER_OWNER | LC_ALL=C sort -u)"

expected_test="$(printf '%s\n' \
    FG_BENCH_EXPERT_DECODE \
    FG_BENCH_GDN_ALGEBRAIC \
    FG_BENCH_HC_INJECT \
    FG_BENCH_KQUANT_COOKED \
    FG_BENCH_PREFILL \
    FG_BENCH_PREFILL_SHAPES \
    FG_BENCH_Q5_1_COOKED \
    FG_BENCH_Q8_COOKED \
    FG_BENCH_Q8_SUBGROUP \
    FG_EXPERT_PARITY_DUMP \
    FG_RMS_HASH | LC_ALL=C sort -u)"

actual_code="$(extract src include)"
actual_test="$(extract tests)"

doc_live="$(tr -d '\r' < "$doc" | awk -F'|' '
    /^## Removed flags/ { removed = 1 }
    !removed && /^\| `FG_/ {
        name = $2
        gsub(/`/, "", name)
        gsub(/^ +| +$/, "", name)
        if (NF == 6) {
            status = $5
            gsub(/^ +| +$/, "", status)
        } else {
            status = "test/bench"
        }
        printf "%s\t%s\n", name, status
    }' "$doc")"
doc_removed="$(tr -d '\r' < "$doc" | awk -F'|' '
    /^## Removed flags/ { removed = 1 }
    removed && /^\| `FG_/ {
        name = $2
        gsub(/`/, "", name)
        gsub(/^ +| +$/, "", name)
        print name
    }' "$doc")"
doc_names="$(printf '%s\n' "$doc_live" | cut -f1 | LC_ALL=C sort -u)"

compare() {
    local label="$1" expected="$2" actual="$3" missing added
    missing="$(comm -23 <(printf '%s\n' "$expected") <(printf '%s\n' "$actual"))"
    added="$(comm -13 <(printf '%s\n' "$expected") <(printf '%s\n' "$actual"))"
    if [[ -n "$missing" ]]; then
        fail "$label missing: $(printf '%s' "$missing" | tr '\n' ' ')"
    fi
    if [[ -n "$added" ]]; then
        fail "$label unexpected: $(printf '%s' "$added" | tr '\n' ' ')"
    fi
}

while IFS=$'\t' read -r name status; do
    [[ -n "$name" ]] || continue
    case "$status" in
        "validated default" | "variant (opt-in)" | "profiling" | "test/bench" | \
            "packer option" | "test/ops") ;;
        *) fail "docs row $name has unknown status '$status'" ;;
    esac
done <<<"$doc_live"

compare "src/include expected set" "$expected_code" "$actual_code"
compare "tests expected set" "$expected_test" "$actual_test"
compare "docs live rows vs code" \
    "$(printf '%s\n%s\n' "$actual_code" "$actual_test" | LC_ALL=C sort -u)" \
    "$doc_names"

while IFS= read -r removed; do
    [[ -n "$removed" ]] || continue
    if printf '%s\n%s\n' "$actual_code" "$actual_test" | grep -qx "$removed"; then
        fail "removed flag $removed is still read by code"
    fi
done <<<"$doc_removed"

code_count="$(printf '%s\n' "$actual_code" | grep -c . || true)"
test_count="$(printf '%s\n' "$actual_test" | grep -c . || true)"
doc_count="$(printf '%s\n' "$doc_names" | grep -c . || true)"
removed_count="$(printf '%s\n' "$doc_removed" | grep -c . || true)"

status=PASS
if [[ "$failures" -ne 0 ]]; then
    status=FAIL
fi
printf 'FGFLAGS code=%s test=%s doc=%s removed=%s status=%s\n' \
    "$code_count" "$test_count" "$doc_count" "$removed_count" "$status"
[[ "$status" == "PASS" ]]
