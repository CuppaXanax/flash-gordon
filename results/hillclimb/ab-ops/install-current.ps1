param([Parameter(Mandatory)][string]$ConfigPath)
$ErrorActionPreference = 'Stop'
$root = 'D:\looking-glass-labs\fg-wt\ab-harness'
$build = Join-Path $root '.build\abq'
$output = Join-Path $root 'results\hillclimb\ab-current'
$runner = Join-Path $root 'tools\ab-fleet.ps1'
$remote = '/home/xander/flash-gordon-live/20261015-abq-operator'
$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json -AsHashtable
if (-not (Test-Path (Join-Path $root 'results\hillclimb\ab-multi-hierarchy.json'))) {
    throw 'Complete qualification before installing the current launcher.'
}
$null = New-Item -ItemType Directory -Force $output

function Fleet([string]$Text, [string]$Tag, [int[]]$Ranks = @(0..7)) {
    $path = Join-Path $build "final-$Tag.sh"
    [IO.File]::WriteAllText($path, $Text.Replace("`r",''))
    $targets = @($Ranks | ForEach-Object { "192.168.42.$(42+$_)" })
    $receipts = @(& $runner -ScriptPath $path -Targets $targets -PassThru -CommandTimeout 180)
    $receipts | ConvertTo-Json -Depth 15 | Set-Content (Join-Path $output "$Tag.json")
    if ($receipts.Count -ne $targets.Count -or @($receipts | Where-Object ExitCode -ne 0).Count) {
        throw "Final fleet $Tag failed. See archived receipts."
    }
}

$helper = (Get-Content "$root\tools\ab-instance.sh" -Raw).Replace("`r",'')
$launcher = @'
#!/usr/bin/env bash
set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
if [[ ${1:-start} == start ]]; then
  printf '%s  %s\n' '__BINARY_SHA__' '__RELEASE__/flash-gordon' | sha256sum -c -
  printf '%s  %s\n' '__MANIFEST_SHA__' '__MANIFEST__' | sha256sum -c -
fi
exec bash "$run/ab-instance.sh" "${1:-start}" '__RELEASE__' '__MANIFEST__' '__SESSIONS__' '__WIDE__'
'@
foreach ($entry in @{
    '__BINARY_SHA__'=$config.binary_sha256; '__MANIFEST_SHA__'=$config.manifest_sha256
    '__RELEASE__'=$config.release; '__MANIFEST__'=$config.manifest
    '__SESSIONS__'=$config.sessions; '__WIDE__'=$config.wide
}.GetEnumerator()) { $launcher = $launcher.Replace($entry.Key,[string]$entry.Value) }
$launcher = $launcher.Replace("`r",'') + "`n"
$json = $config | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText((Join-Path $output 'launch-current.sh'),$launcher)
[IO.File]::WriteAllText((Join-Path $output 'ab-instance.sh'),$helper)
[IO.File]::WriteAllText((Join-Path $output 'current.json'),$json)
$install = @"
set -euo pipefail
run='$remote'
test -d "`$run" && test -O "`$run" && test ! -L "`$run"
test "`$(readlink -f "`$run")" = "`$run"
cd "`$run"
"@
foreach ($item in @(@{name='ab-instance.sh';text=$helper},
                    @{name='launch-current.sh';text=$launcher},
                    @{name='current.json';text=$json})) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($item.text))
    $install += "`nprintf '%s' '$encoded' | base64 -d >'.$($item.name).install'`n"
    $install += "chmod 700 '.$($item.name).install'`nmv '.$($item.name).install' '$($item.name)'`n"
}
$install += "`nbash -n ab-instance.sh`nbash -n launch-current.sh`nsha256sum ab-instance.sh launch-current.sh current.json`n"
Fleet $install install
Fleet "bash '$remote/launch-current.sh' stop" restart-stop
Fleet "bash '$remote/launch-current.sh' start" restart-workers @(1..7)
Fleet "bash '$remote/launch-current.sh' start" restart-coordinator @(0)
$ready = $false
for ($i=0; $i -lt 180; $i++) {
    try {
        if ((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status -eq 'ok') {
            $ready = $true; break
        }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $ready) { throw 'Final persistent launcher API readiness failed.' }
Fleet "bash '$remote/launch-current.sh' status" status
Fleet @'
set -euo pipefail
cd /home/xander/flash-gordon-live/20261015-abq-operator
rank=$(($(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)-42))
python3 - "$rank" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
c=json.loads(Path("current.json").read_text())
rank=int(sys.argv[1])
found=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid != os.getuid():
            continue
        exe=os.readlink(p/"exe")
        if exe.startswith("/home/xander/flash-gordon-live/") and exe.endswith("/flash-gordon"):
            found.append((p,exe))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert len(found)==1, "Expected exactly one serving release process"
p,exe=found[0]
assert exe == c["release"]+"/flash-gordon"
args=[s.decode() for s in (p/"cmdline").read_bytes().split(b"\0") if s]
expected=[exe,"rank","--manifest",c["manifest"],"--rank",str(rank)]
if rank==0:
    expected=[exe,"api","--manifest",c["manifest"],"--model-name",
              "Qwen3.8-Flash-Next-Uncensored","--host","0.0.0.0","--port","8080",
              "--context-tokens","262144","--gpu-index-tokens","262144",
              "--qsa-hot-tokens","0","--qsa-page-cache-mib","700"]
    if c["sessions"]==2:
        expected += ["--sessions","2"]
assert args == expected, "Serving CLI changed"
environment=dict(s.decode().split("=",1) for s in (p/"environ").read_bytes().split(b"\0") if b"=" in s)
assert environment["FG_SHADER_DIR"]==c["release"]+"/vulkan"
assert environment["FG_EXPERT_Q8_WIDE"]==str(c["wide"])
assert rank!=0 or environment["FG_RING_TRACE"]=="1"
assert all(v in ("","0") for k,v in environment.items()
           if k.startswith("FG_") and "PROFILE" in k), "Intrusive profiling is enabled"
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
original_binary=digest("/home/xander/flash-gordon-live/20260926-abl2/flash-gordon")
original_manifest=digest("/home/xander/fg-ring-pack-abliterated/manifest.fgm")
assert original_binary=="e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2"
assert original_manifest=="35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb"
assert digest(p/"exe")==c["binary_sha256"]
assert digest(c["manifest"])==c["manifest_sha256"]
print(json.dumps({"rank":rank,"pid":int(p.name),"args":args,
    "environment":{k:environment[k] for k in ("FG_SHADER_DIR","FG_EXPERT_Q8_WIDE","FG_RING_TRACE") if k in environment},
    "original_binary_sha256":original_binary,"original_manifest_sha256":original_manifest,
    "current_binary_sha256":c["binary_sha256"],"current_manifest_sha256":c["manifest_sha256"]}))
PY
'@ verify
$health = Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 15
$models = Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 15
$chat = Invoke-RestMethod 'http://192.168.42.42:8080/v1/chat/completions' -Method Post `
    -ContentType 'application/json' -TimeoutSec 180 -Body (@{
        model='Qwen3.8-Flash-Next-Uncensored'; temperature=0; max_tokens=16
        messages=@(@{role='user';content='/no_think Output exactly OK.'})
    } | ConvertTo-Json -Depth 6)
if ($models.data[0].id -ne 'Qwen3.8-Flash-Next-Uncensored' -or $chat.usage.completion_tokens -lt 1 -or
    [string]::IsNullOrWhiteSpace($chat.choices[0].message.content)) {
    throw 'Final model or real generation health check failed.'
}
@{verified_at=(Get-Date -Format o);health=$health;models=$models;chat=$chat} |
    ConvertTo-Json -Depth 15 | Set-Content (Join-Path $output 'api-health.json')
Write-Host 'Persistent current launchers installed, restart tested, API/models/chat healthy.'
