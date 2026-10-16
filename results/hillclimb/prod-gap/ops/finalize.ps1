$ErrorActionPreference='Stop'
. "$PSScriptRoot\common.ps1"
& "$PSScriptRoot\promote.ps1"
$null=Fleet @'
set -euo pipefail
python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
operator=Path("/home/xander/flash-gordon-live/20261015-abq-operator")
scratch=Path("/home/xander/hc-abq-gap")
config=json.loads((operator/"current.json").read_text())
rank=int(os.popen("ip -4 addr show | grep -oP '192\\.168\\.42\\.\\K[0-9]+' | head -1").read())-42
pid=int((operator/f"rank-{rank}.pid").read_text())
p=Path("/proc")/str(pid)
environment=dict(s.decode().split("=",1) for s in (p/"environ").read_bytes().split(b"\0") if b"=" in s)
allowed={"FG_SHADER_DIR","FG_EXPERT_Q8_WIDE"}|({"FG_RING_TRACE"} if rank==0 else set())
assert {k for k in environment if k.startswith("FG_")}==allowed, "Unexpected FG flag"
assert "LD_PRELOAD" not in environment
assert config["sessions"]==1 and config["wide"]==0
assert hashlib.sha256((p/"exe").read_bytes()).hexdigest()==config["binary_sha256"]
assert hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()==config["manifest_sha256"]
for q in Path("/proc").iterdir():
    if not q.name.isdigit():
        continue
    try:
        if q.stat().st_uid!=os.getuid():
            continue
        assert str(scratch).encode() not in (q/"cmdline").read_bytes(), f"Scratch argv {q.name}"
        assert str(scratch) not in (q/"maps").read_text(), f"Scratch mapping {q.name}"
        assert not os.readlink(q/"cwd").startswith(str(scratch)), f"Scratch cwd {q.name}"
        for fd in (q/"fd").iterdir():
            try:
                assert not os.readlink(fd).startswith(str(scratch)), f"Scratch fd {q.name}/{fd.name}"
            except (FileNotFoundError,PermissionError):
                pass
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        continue
print(json.dumps({"rank":rank,"pid":pid,"profiling_and_preload_absent":True,
    "environment":{k:environment[k] for k in sorted(allowed)},
    "scratch_unreferenced":True,"current":config}))
PY
scratch=/home/xander/hc-abq-gap
test -d "$scratch" && test -O "$scratch" && test ! -L "$scratch"
test "$(readlink -f "$scratch")" = "$scratch"
rm -rf -- "$scratch"
test ! -e "$scratch"
printf '%s\n' 'Experiment scratch removed; pinned mix-04 remains live.'
'@ 'final-cleanup'
$health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 15
$models=Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 15
$chat=Invoke-RestMethod 'http://192.168.42.42:8080/v1/chat/completions' -Method Post -ContentType 'application/json' -TimeoutSec 180 -Body (@{
    model='Qwen3.8-Flash-Next-Uncensored';temperature=0;top_p=1;max_tokens=16
    messages=@(@{role='user';content='/no_think Output exactly OK.'})
}|ConvertTo-Json -Depth 6)
if($health.status-ne'ok'-or$models.data[0].id-ne'Qwen3.8-Flash-Next-Uncensored'-or
   [string]::IsNullOrWhiteSpace($chat.choices[0].message.content)){throw 'Post-cleanup API check failed'}
@{verified_at=(Get-Date -Format o);health=$health;models=$models;chat=$chat}|
    ConvertTo-Json -Depth 15|Set-Content "$output\final-api-health.json"
Write-Host 'Final plain mix-04 restart, exact identities, trace-off, cleanup and API checks passed.'
