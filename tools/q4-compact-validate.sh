set -euo pipefail
cd /home/xander/hc-q4
mkdir -p compact
cd compact
export MESA_SHADER_CACHE_DIR="$PWD/.mesa-cache"
if [ "${FG_Q4_COMPACT_SAFE:-0}" != 1 ]; then
  tar -xf ../compact.tar
  sed -i 's/\r$//' tools/q4-safe-tests.sh tools/q4-compact-validate.sh
  export FG_Q4_COMPACT_SAFE=1
  exec bash tools/q4-safe-tests.sh bash tools/q4-compact-validate.sh
fi
make -j8 flash-gordon tools/q4-repack tests/test_core tests/test_q4_repack tests/test_model_load tests/test_fg_vk > validation-build.log 2>&1 || { tail -65 validation-build.log; exit 1; }
./tests/test_core > core-validation.log 2>&1
tail -2 core-validation.log
./tests/test_q4_repack
for fixture in .q4-repack-source.bin .q4-repack-cooked.bin; do
  test ! -e "$fixture"
  printf 'preserve-existing-fixture\n' > "$fixture"
  if ./tests/test_q4_repack > exclusive-fixture.log 2>&1; then
    echo 'Expected existing-fixture refusal'; exit 1
  fi
  test "$(cat "$fixture")" = preserve-existing-fixture
  rm -- "$fixture"
done
echo 'Q4 file-cooker exclusive fixture preservation: PASS'
./tests/test_model_load
for filter in dense_quant_compact dense_quant_soa q8_dense_cooked q8_cooked_prefill_parity; do
  DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk
done
