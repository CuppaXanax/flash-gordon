#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-reconcile
test -f complete
python3 - <<'PY'
from pathlib import Path
import hashlib,json
rows=json.loads(Path('evidence/positions.json').read_text())
directory=Path('final-hyper');directory.mkdir(exist_ok=True)
receipts={}
for path in sorted(Path('pack').glob('*-current.candidate')):
    output=directory/(path.name.removesuffix('-current.candidate')+'.f32')
    with path.open('rb') as source,output.open('wb') as target:
        for row in rows:
            source.seek((row['tokens']-1)*40960,1)
            data=source.read(40960)
            assert len(data)==40960
            target.write(data)
        assert source.read(1)==b''
    with output.open('rb') as f:
        receipts[output.name]=dict(bytes=output.stat().st_size,
            sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert len(receipts)==6
Path('final-hyper/identity.json').write_text(json.dumps(receipts,indent=2)+'\n')
PY
tar -czf reconcile-prepared560.tar.gz pack/cases.bin pack/input.teacher pack/ngram.f32 \
    corpus.json prompts.fgev evidence/prepared.sha256 evidence/positions.json
tar -czf reconcile-final-hyper.tar.gz final-hyper
sha256sum reconcile-prepared560.tar.gz reconcile-final-hyper.tar.gz
