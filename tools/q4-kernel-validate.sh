set -euo pipefail
cd /home/xander/hc-q4/kernels
export MESA_SHADER_CACHE_DIR="$PWD/.mesa-cache"
if [ "${FG_Q4_SAFE_VALIDATION:-0}" != 1 ]; then
  tar -xf ../kernels.tar
  sed -i 's/\r$//' tools/q4-safe-tests.sh tools/q4-kernel-validate.sh
  export FG_Q4_SAFE_VALIDATION=1
  exec bash tools/q4-safe-tests.sh bash tools/q4-kernel-validate.sh
fi
make -j8 tests/test_fg_vk tests/test_core tests/test_model_load tools/q4-repack flash-gordon > validation-build.log 2>&1 || { tail -70 validation-build.log; exit 1; }
unset FG_BENCH_DENSE_SOA FG_BENCH_DENSE_QUANT
./tests/test_core > core-validation.log 2>&1
tail -4 core-validation.log
./tests/test_model_load
for filter in dense_quant dense_quant_cooked dense_quant_soa q8_dense q8_dense_cooked q8_cooked_prefill_parity q8_decode_shape_parity q8_split_silu_parity router_bf16_exact; do
  DS4_REMOTE_TEST_FILTER=$filter ./tests/test_fg_vk 2>&1
done
set +e
./tools/q4-repack missing.fgm missing-pack missing.gguf unused 0 --keep-q8-head
status=$?
set -e
test "$status" -eq 1
printf 'Q8-head option parsed; missing manifest correctly refused.\n'
vulkaninfo 2>/dev/null | grep -E 'shaderEngineCount|shaderArraysPerEngineCount|computeUnitsPerShaderArray|activeComputeUnitCount' || true
