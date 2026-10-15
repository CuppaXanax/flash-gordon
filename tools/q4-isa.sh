#!/bin/bash
set -euo pipefail
cd /home/xander/hc-q4/code
MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders DS4_REMOTE_TEST_FILTER=dense_quant ./tests/test_fg_vk >../shader-isa.log 2>&1
grep -E 'SGPR|VGPR|Spill|Occup|Wave|wave|registers|scratch|Code size|code size|s_endpgm' ../shader-isa.log | tail -40
