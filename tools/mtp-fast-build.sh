#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
python3 - <<'PY'
from pathlib import Path
import tarfile
root = Path("source").resolve()
with tarfile.open("fast-source.tar") as archive:
    for member in archive:
        path = root / member.name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"unsafe source member: {member.name}")
        if member.isdir():
            path.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            data = archive.extractfile(member).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)
            path.chmod(member.mode & 0o777)
        else:
            raise ValueError(f"unsupported source member: {member.name}")
PY
cd source
mkdir -p .build/compiler
export TMPDIR="$PWD/.build/compiler" XDG_CACHE_HOME="$PWD/.build/cache"
make -j8 flash-gordon tools/bench-mtp-head tests/test_mtp tests/test_sampler tests/test_decode_batch tests/test_fg_vk \
    > .build/fast-build.log 2>&1 || { tail -70 .build/fast-build.log; exit 1; }
./tests/test_mtp
./tests/test_sampler
./tests/test_decode_batch
bash tools/check-flags.sh
./tools/bench-mtp-head --self-test
python3 tests/test_mtp_quantize.py
: > .build/fast-tests.log
for filter in static_batch_replay q8_dense_cooked dense_quant_compact dense_quant_soa \
    expert_decode_fused expert_decode_fused_q8_0 expert_decode_fused_q8_0_cooked expert_decode_fused_q8_gates; do
    DS4_REMOTE_TEST_FILTER="$filter" ./tests/test_fg_vk >> .build/fast-tests.log 2>&1 \
        || { tail -40 .build/fast-tests.log; exit 1; }
done
tail -12 .build/fast-tests.log
