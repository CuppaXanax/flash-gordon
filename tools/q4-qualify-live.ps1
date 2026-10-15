#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$SourceArchive,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidatePattern('^/home/xander/flash-gordon-live/[0-9]{8}-q4[-a-z0-9]*$')]
    [string]$Release='/home/xander/flash-gordon-live/20261015-q4-ordinary',
    [ValidatePattern('^[a-z0-9-]+$')][string]$Attempt='r1'
)
$ErrorActionPreference='Stop'
$SourceArchive=(Get-Item -LiteralPath $SourceArchive).FullName
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$root=Split-Path $PSScriptRoot
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$helper=Join-Path $PSScriptRoot 'decode-instance.ps1'
$harness='D:\looking-glass-labs\flash-gordon\tools\hillclimb.ps1'
$harnessRuns='D:\looking-glass-labs\flash-gordon\results\hillclimb\runs'
$original='/home/xander/fg-ring-pack-abliterated/manifest.fgm'
$teacher='/home/xander/fg-ring-pack-abliterated-q4-teacher/manifest.fgm'
$ordinary='/home/xander/fg-ring-pack-abliterated-q4/manifest.fgm'
$protected='/home/xander/fg-ring-pack-abliterated-q4-protected/manifest.fgm'
$originalRelease='/home/xander/flash-gordon-live/20260926-abl2'
$targets=@(42..49 | ForEach-Object { "192.168.42.$_" })
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$command=Join-Path $OutputDir "live-command-$PID.sh"
$promoted=$false
$teacherTag="$Attempt-teacher"
function Fleet([string]$Text,[string]$Receipt) {
    [IO.File]::WriteAllText($command,$Text.Replace("`r",''))
    $records=@(& $fleet -ScriptPath $command -Targets $targets -PassThru)
    $records | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutputDir "$Receipt.json")
    if ($records.Count -ne 8 -or @($records | Where-Object ExitCode -ne 0).Count) {
        throw "Fleet operation failed: $Receipt"
    }
    return $records
}
function Instance([string]$Action,[string]$Manifest,[string]$Mode,[string]$Tag,
    [int[]]$Ranks=@(0..7),[string]$Binary=$Release) {
    & $helper -Action $Action -Release $Binary -Manifest $Manifest -Mode $Mode `
      -CaptureTag $Tag -Ranks $Ranks -FleetScript $fleet
}
function Stop-All {
    foreach ($manifest in @($original,$teacher,$ordinary,$protected)) {
        Instance stop $manifest plain stop
    }
}
function Wait-Api {
    for ($i=0; $i -lt 120; $i++) {
        try {
            $health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2
            if ($health.status -eq 'ok') { return }
        } catch {}
        Start-Sleep -Seconds 3
    }
    throw 'API readiness timeout'
}
function Restore-Original {
    Write-Host 'Restoring the immutable original release and manifest'
    Stop-All
    Instance start $original plain restored @(0..7) $originalRelease
    Wait-Api
}
function Prepare([string]$Label,[string]$Manifest) {
    $text=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'q4-qualify-ready.sh'))
    $text=$text.Replace('label=${1:?label}',"label='$Label'")
    $text=$text.Replace('pack=${2:?pack}',"pack='$($Manifest.Substring(0,$Manifest.LastIndexOf('/')))'")
    $text=$text.Replace('release=${3:?release}',"release='$Release'")
    $records=Fleet $text "$Label-ready"
    $hashes=@($records | ForEach-Object {
        $m=[regex]::Match($_.Stdout,'(?m)^([a-f0-9]{64})  '+[regex]::Escape("$Release/flash-gordon")+'$')
        if(-not $m.Success){throw 'Missing candidate binary digest'}
        $m.Groups[1].Value
    } | Sort-Object -Unique)
    if($hashes.Count -ne 1){throw 'Owner binaries differ'}
}
function Capture([string]$Tag,[string]$Manifest) {
    Stop-All
    Instance start $Manifest quality $Tag @(1..7)
    & (Join-Path $PSScriptRoot 'q4-capture.ps1') -BuildDir $Release -Manifest $Manifest `
      -OutputDir (Join-Path $OutputDir $Tag) -ResidentBatch -CaptureTag $Tag
    Stop-All
}
function Harness([string]$Tag,[switch]$Quick) {
    $before=Get-Date
    $log=Join-Path $OutputDir "$Tag-harness.log"
    $args=@('-NoProfile','-File',$harness,'-BaseUrl','http://192.168.42.42:8080/v1')
    if($Quick){$args+='-Quick'}
    & pwsh @args *> $log
    $code=$LASTEXITCODE
    $file=Get-ChildItem $harnessRuns -Filter '*.json' | Where-Object LastWriteTime -ge $before |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if(-not $file){throw "No harness record: $Tag"}
    $destination=Join-Path $OutputDir "$Tag-harness.json"
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    if($code -notin @(0,2)){throw "Harness failed, exit=$code; record=$destination"}
    return Get-Content -LiteralPath $destination -Raw | ConvertFrom-Json
}
function Start-Api([string]$Manifest,[string]$Tag) {
    Stop-All
    Instance start $Manifest plain $Tag
    Wait-Api
    $text=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'q4-qualify-resident.sh'))
    $text=$text.Replace('manifest=${1:?manifest}',"manifest='$Manifest'")
    Fleet $text "$Attempt-$Tag-resident" | Out-Null
}
try {
    Prepare qualify-ordinary $ordinary
    Capture $teacherTag $teacher
    Start-Api $teacher baseline
    $control=Harness "$Attempt-baseline" -Quick
    if(-not $control.metadata.correctness_pass){throw 'Same-binary Q8 control changes the greedy reference'}
    foreach($plan in @('ordinary','protected')){
        if($plan -eq 'protected'){
            Restore-Original
            $cleanup=@'
set -euo pipefail
for pack in /home/xander/fg-ring-pack-abliterated-q4 /home/xander/fg-ring-pack-abliterated-q4-teacher; do
  find "$pack" -maxdepth 1 -type f -name 'qsa-*.state*' -delete
done
df -B1 /home/xander
'@
            Fleet $cleanup 'before-protected' | Out-Null
            & (Join-Path $PSScriptRoot 'q4-stage-pack.ps1') -SourceArchive $SourceArchive `
              -OutputDir (Join-Path $OutputDir 'protected-stage') -Label qualify-protected `
              -OverlayName dense-protected -AfterLease
            Prepare qualify-protected $protected
        }
        $manifest=if($plan -eq 'ordinary'){$ordinary}else{$protected}
        $tag="$Attempt-$plan"
        Capture $tag $manifest
        $quality=Join-Path $OutputDir "$tag-quality.json"
        & python (Join-Path $PSScriptRoot 'q4_quality.py') --teacher (Join-Path $OutputDir $teacherTag) `
          --candidate (Join-Path $OutputDir $tag) --output $quality
        if($LASTEXITCODE -notin @(0,1)){throw 'Quality comparison execution failed'}
        Start-Api $manifest $plan
        $run=Harness $tag
        & python (Join-Path $PSScriptRoot 'q4_qualify.py') `
          --baseline 'D:\looking-glass-labs\flash-gordon\results\hillclimb\BASELINE.json' `
          --run (Join-Path $OutputDir "$tag-harness.json") --quality $quality `
          --output (Join-Path $OutputDir "$tag-gates.json")
        if($LASTEXITCODE -notin @(0,1)){throw 'Qualification comparison execution failed'}
        $gates=Get-Content (Join-Path $OutputDir "$tag-gates.json") -Raw | ConvertFrom-Json
        if($gates.passed){
            $promoted=$true
            @{promoted=$plan; release=$Release; manifest=$manifest} | ConvertTo-Json |
              Set-Content (Join-Path $OutputDir 'endstate.json')
            break
        }
    }
} finally {
    if(-not $promoted){
        Restore-Original
        @{promoted=$false; release=$originalRelease; manifest=$original} | ConvertTo-Json |
          Set-Content (Join-Path $OutputDir 'endstate.json')
    }
    Remove-Item -LiteralPath $command -ErrorAction SilentlyContinue
}
