#requires -Version 7.0
param(
    [Parameter(Mandatory)][hashtable]$Deployment,
    [string]$Arm = '',
    [switch]$SnapshotOnly,
    [string]$FleetScript = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$operator = Join-Path $root '.build\abq'
$null = New-Item -ItemType Directory -Force $operator
$statePath = Join-Path $operator 'live.json'
$original = @{
    release = '/home/xander/flash-gordon-live/20260926-abl2'
    manifest = '/home/xander/fg-ring-pack-abliterated/manifest.fgm'
    manifest_sha256 = '35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb'
    binary_sha256 = 'e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2'
    sessions = 1; wide = 0
}
$live = if (Test-Path $statePath) { Get-Content $statePath -Raw | ConvertFrom-Json -AsHashtable } else { $original }
function Invoke-ABFleet {
    param([string]$Action, [hashtable]$Config, [int[]]$Ranks)
    $source = Get-Content -Raw (Join-Path $PSScriptRoot 'ab-instance.sh')
    foreach ($key in @('release','manifest','sessions','wide')) {
        if (-not $Config.ContainsKey($key)) { throw "Missing deployment $key" }
    }
    $source = $source.Replace('action=${1:?action}', "action='$Action'").
        Replace('release=${2:?release}', "release='$($Config.release)'").
        Replace('manifest=${3:?manifest}', "manifest='$($Config.manifest)'").
        Replace('sessions=${4:-1}', "sessions='$($Config.sessions)'").
        Replace('wide=${5:-0}', "wide='$($Config.wide)'")
    if ($Action -eq 'start') {
        $check = "printf '%s  %s\n' '$($Config.binary_sha256)' '$($Config.release)/flash-gordon' | sha256sum -c -`n" +
            "printf '%s  %s\n' '$($Config.manifest_sha256)' '$($Config.manifest)' | sha256sum -c -`n"
        $source = $source.Replace('case "$action" in', $check + 'case "$action" in')
    }
    $script = Join-Path $operator "operator-$PID.sh"
    [IO.File]::WriteAllText($script, $source.Replace("`r",''))
    try {
        $targets = @($Ranks | ForEach-Object { "192.168.42.$(42+$_)" })
        $records = @(& $FleetScript -ScriptPath $script -Targets $targets -PassThru -CommandTimeout 180)
        $records | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $operator ("fleet-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + "-$Action.json"))
        if ($records.Count -ne $targets.Count -or @($records | Where-Object ExitCode -ne 0).Count) {
            $records | ForEach-Object { Write-Host $_.IP $_.Stdout $_.Stderr }
            throw "Fleet $Action failed."
        }
    } finally { Remove-Item $script -Force -ErrorAction SilentlyContinue }
}
if ($SnapshotOnly) {
    Invoke-ABFleet status $Deployment @(0..7)
    return
}
Invoke-ABFleet stop $live @(0..7)
try {
    # Persist attempted identity before starting so rollback can stop partial launches.
    $Deployment | ConvertTo-Json | Set-Content $statePath
    Invoke-ABFleet start $Deployment @(1..7)
    Invoke-ABFleet start $Deployment @(0)
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        try {
            if ((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status -eq 'ok') {
                $ready = $true; break
            }
        } catch {}
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'API readiness timeout.' }
    Invoke-ABFleet status $Deployment @(0..7)
    Write-Host "AB_DEPLOY arm=$Arm release=$($Deployment.release) sessions=$($Deployment.sessions) wide=$($Deployment.wide) healthy=true"
} catch {
    throw
}
