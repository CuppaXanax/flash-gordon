#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /home/xander/hc-q4/code
full_rc=0
bash tools/q4-safe-tests.sh make -j8 test test-vulkan >../regression.log 2>&1 || full_rc=$?
tail -30 ../regression.log
if [ "$full_rc" != 0 ]; then
    echo "Full suite blocked (exit $full_rc); run standalone core and native Vulkan oracles."
    bash tools/q4-safe-tests.sh make -j8 tests/test_core tests/test_fg_vk >../standalone-build.log 2>&1
    ./tests/test_core >../standalone-core.log 2>&1
    ./tests/test_fg_vk >../standalone-vulkan.log 2>&1 || { tail -80 ../standalone-vulkan.log; exit 1; }
    tail -5 ../standalone-core.log
    tail -12 ../standalone-vulkan.log
fi
python3 -m unittest tools/test_q4_tools.py
exit "$full_rc"
