#requires -Version 7.0
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$build = Join-Path $root '.build\rollback-main-mix04'
$null = New-Item -ItemType Directory -Force $build
$remote = '/home/xander/flash-gordon-live/20261015-abq-operator'
$script:rollbackStep = 0
function Invoke-RollbackStep([string]$Text, [int[]]$Ranks) {
    $path = Join-Path $build 'step.sh'
    [IO.File]::WriteAllText($path, $Text.Replace("`r",''))
    $targets = @($Ranks | ForEach-Object { "192.168.42.$(42+$_)" })
    $r = @(& "$PSScriptRoot\ab-fleet.ps1" -ScriptPath $path -Targets $targets -PassThru -CommandTimeout 180)
    $r | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $build "$script:rollbackStep.json")
    $script:rollbackStep++
    if ($r.Count -ne $targets.Count -or @($r | Where-Object ExitCode -ne 0).Count) {
        throw 'Main/mix-04 rollback failed; inspect owned-PID state and step receipts.'
    }
}
Invoke-RollbackStep @"
set -euo pipefail
cd '$remote'
test -f launch-main-mix04.sh && test ! -L launch-main-mix04.sh
test -f main-mix04.json && test ! -L main-mix04.json
bash launch-current.sh stop
cp launch-main-mix04.sh launch-current.sh
cp main-mix04.json current.json
"@ @(0..7)
Invoke-RollbackStep "bash '$remote/launch-current.sh' start" @(1..7)
Invoke-RollbackStep "bash '$remote/launch-current.sh' start" @(0)
$ready = $false
for ($i=0; $i -lt 180; $i++) {
    try {
        if ((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status -eq 'ok') {
            $ready = $true; break
        }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $ready) { throw 'Main/mix-04 API readiness timed out.' }
Invoke-RollbackStep "bash '$remote/launch-current.sh' status" @(0..7)
$baseline = Get-Content "$root\results\hillclimb\prod-gap\current\current.json" -Raw |
    ConvertFrom-Json -AsHashtable
$null = New-Item -ItemType Directory -Force (Join-Path $root '.build\abq')
$baseline | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $root '.build\abq\live.json')
$models = Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 10
$chat = Invoke-RestMethod 'http://192.168.42.42:8080/v1/chat/completions' -Method Post `
    -ContentType 'application/json' -TimeoutSec 180 -Body (@{
        model='Qwen3.8-Flash-Next-Uncensored'; temperature=0; max_tokens=16
        messages=@(@{role='user';content='/no_think Output exactly OK.'})
    } | ConvertTo-Json -Depth 5)
if ($models.data[0].id -ne 'Qwen3.8-Flash-Next-Uncensored' -or
    [string]::IsNullOrWhiteSpace($chat.choices[0].message.content)) {
    throw 'Main/mix-04 model/generation verification failed.'
}
@{models=$models;chat=$chat;verified=[DateTimeOffset]::Now.ToString('o')} |
    ConvertTo-Json -Depth 12 | Set-Content (Join-Path $build 'api-health.json')
Write-Host 'Main a63a166/mix-04 sessions 1 restored and pinned; health/models/greedy chat verified.'
