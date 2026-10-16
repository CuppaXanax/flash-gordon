#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
sudo -n prlimit --pid "$$" --memlock=268435456:268435456
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
PYTHONPATH=source/tools python3 - <<'PY'
from mtp_source import Source,get_json
from mtp_real_tensor import REPO,REVISION
from pathlib import Path
import hashlib,json,struct
s=Source(REPO,REVISION)
items=[x for x in get_json(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}/UD-Q4_K_XL')
       if x['path'].endswith('.gguf')]
item=items[0]
raw=s.read_range(item['path'],0,min(item['size'],16*1024*1024))
Path('pack/public-header.gguf').write_bytes(raw)
proof=dict(repo=REPO,revision=REVISION,shard=item['path'],
           prefix_bytes=len(raw),prefix_sha256=hashlib.sha256(raw).hexdigest(),
           tensors=struct.unpack_from('<Q',raw,8)[0])
print(json.dumps(proof))
Path('tokenizer-source.json').write_text(json.dumps(proof,indent=2)+'\n')
PY
source/tools/bf16-offline tokenizer pack/manifest.fgm pack pack/public-header.gguf \
    > tokenizer-check.log
echo 'b291b98670183858d414610cfe979e77a67ca53a1b0e977ed568548f8558c820  pack/tokenizer/tokenizer.fgt' | sha256sum -c -
