set -euo pipefail
op=/home/xander/flash-gordon-live/20261015-abq-operator
test -f "$op/lease-active.json"
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/home/xander/flash-gordon-live/20261015-abq-operator')
r=json.loads((p/'lease-baseline.json').read_text())
assert r['binary_sha256']=='02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0'
assert r['release']=='/home/xander/flash-gordon-live/20261016-abq-stack-adaptive'
assert r['sessions']==2
for source,target in [('launch-lease-baseline.sh','launch-pre-robustness-20261018.sh'),
                      ('lease-baseline.json','pre-robustness-20261018.json')]:
    content=(p/source).read_bytes()
    if (p/target).exists():
        assert (p/target).read_bytes()==content
    else:
        (p/target).write_bytes(content)
print('Durable exact stack-1 rollback helpers verified')
PY
sha256sum "$op/launch-pre-robustness-20261018.sh" "$op/pre-robustness-20261018.json"
