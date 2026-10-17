#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-reconcile
find . -maxdepth 1 -name '*.complete' -o -name '*.ready' | sort
df -B1 /home/xander | tail -n 1
for f in pack/ngram.iq4nl pack/rank-00.mixed.fgw pack/rank-01.mixed.fgw; do
    [[ ! -f "$f" ]] || stat -c '%n %s' "$f"
done
[[ ! -f evidence/ngram-source.log ]] || tail -n 2 evidence/ngram-source.log
for f in evidence/*stage*.jsonl; do
    [[ -f "$f" ]] || continue
    printf '%s ' "$f"
    tail -n 1 "$f"
done
for f in evidence/*stage*.stderr; do
    [[ -f "$f" ]] || continue
    printf '%s\n' "$f"
    tail -n 2 "$f"
done
python3 - <<'PY'
from pathlib import Path
for p in Path('/proc').glob('[0-9]*'):
    try:
        if str(p.joinpath('cwd').resolve())!='/home/xander/hc-reconcile':
            continue
        print('OWNED',p.name,p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()[:200])
    except (PermissionError,FileNotFoundError,ProcessLookupError):
        pass
PY
for service in bc250-embed bc250-rerank; do
    printf '%s ' "$service"
    systemctl is-active "$service" || true
done
