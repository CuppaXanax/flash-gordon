set -euo pipefail
cd /home/xander/hc-q4/kernels
MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders DS4_REMOTE_TEST_FILTER=dense_quant_soa ./tests/test_fg_vk > isa-final.txt 2>&1
grep -nE 'scratch_load|scratch_store|Spill|VGPR|SGPR|disasm:' isa-final.txt | tail -60 || true
wc -l isa-final.txt
gzip -c isa-final.txt > isa-final.txt.gz
