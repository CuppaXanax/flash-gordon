#requires -Version 7.0
[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$build=Join-Path $root '.build\rollback-pre-stack'
$null=New-Item -ItemType Directory -Force $build
$remote='/home/xander/flash-gordon-live/20261015-abq-operator'
$step=0
function Invoke-PreStackStep([string]$Text,[int[]]$Ranks) {
    $script=Join-Path $build 'command.sh'
    [IO.File]::WriteAllText($script,$Text.Replace("`r",''))
    $targets=@($Ranks|ForEach-Object{"192.168.42.$(42+$_)"})
    $r=@(& "$PSScriptRoot\ab-fleet.ps1" -ScriptPath $script -Targets $targets -PassThru -CommandTimeout 180)
    $r|ConvertTo-Json -Depth 10|Set-Content (Join-Path $build "$script:step.json")
    $script:step++
    if($r.Count-ne$targets.Count-or@($r|Where-Object ExitCode -ne 0).Count){
        throw 'Pre-stack rollback failed; inspect owned-PID and step receipts.'
    }
}
Invoke-PreStackStep @"
set -euo pipefail
cd '$remote'
test -f launch-pre-stack-20261016.sh && test ! -L launch-pre-stack-20261016.sh
test -f pre-stack-20261016.json && test ! -L pre-stack-20261016.json
python3 -c 'import json; d=json.load(open("pre-stack-20261016.json")); assert d["release"]=="/home/xander/flash-gordon-live/20261016-abq-multisess" and d["sessions"]==2 and d["wide"]==0 and d["binary_sha256"]=="de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58" and d["manifest_sha256"]=="0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252"'
bash launch-current.sh stop
cp launch-pre-stack-20261016.sh launch-current.sh
cp pre-stack-20261016.json current.json
"@ @(0..7)
Invoke-PreStackStep "bash '$remote/launch-current.sh' start" @(1..7)
Invoke-PreStackStep "bash '$remote/launch-current.sh' start" @(0)
$ready=$false
for($i=0;$i-lt180;$i++){
    try{if((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status-eq'ok'){$ready=$true;break}}catch{}
    Start-Sleep -Seconds 2
}
if(-not $ready){throw 'Pre-stack API readiness timed out.'}
Invoke-PreStackStep "bash '$remote/launch-current.sh' status" @(0..7)
$A=@{
    release='/home/xander/flash-gordon-live/20261016-abq-multisess'
    manifest='/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm'
    binary_sha256='de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58'
    manifest_sha256='0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252'
    shader_tree_sha256='cf403d9d6a68c7cf9b9304af2937365fd210a91b12c52039c1536eed2a1d22b6'
    sessions=2;wide=0
}
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($A|ConvertTo-Json -Depth 10)))
$verify=Get-Content "$PSScriptRoot\stack-verify-selected.sh" -Raw
Invoke-PreStackStep ("set -- '$encoded'`n"+$verify) @(0..7)
$null=New-Item -ItemType Directory -Force (Join-Path $root '.build\abq')
$A|ConvertTo-Json|Set-Content (Join-Path $root '.build\abq\live.json')
$models=Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 10
$chat=Invoke-RestMethod 'http://192.168.42.42:8080/v1/chat/completions' -Method Post `
    -ContentType 'application/json' -TimeoutSec 180 -Body (@{
        model='Qwen3.8-Flash-Next-Uncensored';temperature=0;max_tokens=16;user='rollback-check'
        messages=@(@{role='user';content='/no_think Output exactly OK.'})
    }|ConvertTo-Json -Depth 5)
if($models.data[0].id-ne'Qwen3.8-Flash-Next-Uncensored'-or
    [string]::IsNullOrWhiteSpace($chat.choices[0].message.content)){
    throw 'Pre-stack model/greedy generation verification failed.'
}
@{models=$models;chat=$chat;verified=[DateTimeOffset]::Now.ToString('o')}|
    ConvertTo-Json -Depth 12|Set-Content (Join-Path $build 'api-health.json')
Write-Host 'Adaptive 2d0f8a7/mix-04 sessions2 restored and pinned; health/models/greedy verified.'
