#!/bin/bash
set -euo pipefail
# Redirect legacy test fixtures without changing the tracked test sources.
export TMPDIR="$(pwd)/.q4-test-files"
mkdir -p "$TMPDIR"
restore() {
    for backup in tests/*.q4-saved; do
        [ -e "$backup" ] || continue
        mv "$backup" "${backup%.q4-saved}"
    done
}
trap restore EXIT
python3 - <<'PY'
import os
from pathlib import Path
root = os.environ["TMPDIR"].encode() + b"/"
for path in Path("tests").glob("*.c"):
    original = path.read_bytes()
    patched = original.replace(b"/tmp/", root)
    if patched != original:
        backup = path.with_name(path.name+".q4-saved")
        if backup.exists():
            raise RuntimeError(f"unfinished test source backup: {backup}")
        backup.write_bytes(original)
        path.write_bytes(patched)
PY
"$@"
