#requires -Version 7.0
param(
    [Parameter(Mandatory)][ValidateSet('stop','start','logs','status')][string]$Action,
    [Parameter(Mandatory)][ValidatePattern('^/home/xander/flash-gordon-live/([0-9]{8}-q4[-a-zA-Z0-9]*|20260926-abl2)$')][string]$Release,
    [Parameter(Mandatory)][string]$FleetScript,
    [ValidatePattern('^/home/xander/fg-ring-pack-abliterated(-q4(-protected|-teacher|-control|-ablate-[a-z0-9-]+|-mixed)?)?/manifest\.fgm$')]
    [string]$Manifest='/home/xander/fg-ring-pack-abliterated/manifest.fgm',
    [ValidateSet('plain','trace','profile','prefill','quality','numerics')][string]$Mode='plain',
    [ValidatePattern('^[a-z0-9-]+$')][string]$CaptureTag='run',
    [ValidateRange(0,7)][int[]]$Ranks=@(0..7)
)
$ErrorActionPreference='Stop'
$output=Join-Path (Split-Path $PSScriptRoot) 'results\hillclimb\q4-qualification'
New-Item -ItemType Directory -Force $output | Out-Null
$text=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'decode-instance.sh'))
$text=$text.Replace('action=${1:?action}',"action='$Action'").Replace('release=${2:?release}',"release='$Release'")
$text=$text.Replace('mode=${3:-plain}',"mode='$Mode'")
$text=$text.Replace('manifest=${4:-/home/xander/fg-ring-pack-abliterated/manifest.fgm}',"manifest='$Manifest'")
$text=$text.Replace('tag=${5:-run}',"tag='$CaptureTag'")
$path=Join-Path $output "q4-operator-$PID.sh"
[IO.File]::WriteAllText($path,$text.Replace("`r",''))
try {
    $targets=@($Ranks | Sort-Object -Unique | ForEach-Object { "192.168.42.$(42+$_)" })
    if (-not $targets.Count) { throw 'At least one rank is required' }
    $records=@(& $FleetScript -ScriptPath $path -Targets $targets -CommandTimeout 180 -PassThru)
    $records | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $output "operator-$Action-$Mode-$CaptureTag.json")
    foreach ($record in $records) {
        if ($Action -eq 'logs' -or $record.ExitCode -ne 0 -or
            -not [string]::IsNullOrWhiteSpace($record.Stdout) -or
            -not [string]::IsNullOrWhiteSpace($record.Stderr)) {
            Write-Host "[$($record.IP)] $($record.ExitCode)"
            if ($Action -ne 'logs') { Write-Host $record.Stdout $record.Stderr }
        }
    }
    if ($records.Count -ne $targets.Count -or @($records | Where-Object ExitCode -ne 0).Count) {
        throw 'Fleet lifecycle action failed'
    }
} finally {
    Remove-Item -LiteralPath $path -Force
}
