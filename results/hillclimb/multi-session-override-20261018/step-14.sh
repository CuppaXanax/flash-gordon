set -euo pipefail
python3 - <<'PY'
import os,signal
from pathlib import Path
op=Path('/home/xander/flash-gordon-live/20261015-abq-operator')
pid=int((op/'rank-0.pid').read_text())
assert os.readlink(f'/proc/{pid}/exe')=='/home/xander/flash-gordon-live/20261018-abq-session-recovery-r2/flash-gordon'
assert b'/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm' in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
os.kill(pid,signal.SIGKILL)
print(pid)
PY