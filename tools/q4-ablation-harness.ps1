#requires -Version 7.0
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^/home/xander/fg-ring-pack-abliterated-q4-ablate-(mix-[a-z0-9-]+|aligned-control)/manifest\.fgm$')]
    [string]$Manifest,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ManifestSHA,
    [Parameter(Mandatory)][string]$Quality,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9-]+$')][string]$Tag,
    [Parameter(Mandatory)][string]$OutputDir,
    [switch]$PromoteOnPass
)
$ErrorActionPreference='Stop'
if($PromoteOnPass -and $Manifest.EndsWith('-aligned-control/manifest.fgm')){
    throw 'Q8 performance control is diagnostic only'
}
$qualityRecord=Get-Content $Quality -Raw|ConvertFrom-Json
if(-not $qualityRecord.passed -or $qualityRecord.samples -lt 400 -or $qualityRecord.full_vocab -ne 248320){
    throw 'Require a passing full-vocabulary quality capture with at least 400 frontiers'
}
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir|Out-Null
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$helper=Join-Path $PSScriptRoot 'decode-instance.ps1'
$release='/home/xander/flash-gordon-live/20261015-q4-ablation'
$original='/home/xander/fg-ring-pack-abliterated/manifest.fgm'
$originalRelease='/home/xander/flash-gordon-live/20260926-abl2'
$mainRoot='D:\looking-glass-labs\flash-gordon'
$command=Join-Path $OutputDir "$Tag-harness-command.sh"
$targets=@(42..49|ForEach-Object{"192.168.42.$_"})
$promoted=$false
$endStateReady=$false
function Instance([string]$Action,[string]$Path,[string]$Binary=$release){
    & $helper -Action $Action -Manifest $Path -Release $Binary -Mode plain `
      -CaptureTag $Tag -Ranks @(0..7) -FleetScript $fleet
}
function Ready {
    for($i=0;$i -lt 120;$i++){
        try{if((Invoke-RestMethod http://192.168.42.42:8080/health -TimeoutSec 2).status -eq 'ok'){return}}catch{}
        Start-Sleep -Seconds 3
    }
    throw 'Harness API readiness timeout'
}
function Fleet([string]$Text,[string]$Name){
    [IO.File]::WriteAllText($command,$Text.Replace("`r",''))
    $r=@(& $fleet -ScriptPath $command -Targets $targets -PassThru)
    $r|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputDir "$Tag-$Name.json")
    if($r.Count -ne 8 -or @($r|Where-Object ExitCode -ne 0).Count){throw "Fleet operation failed: $Name"}
    return $r
}
try{
    $check=@'
set -e
printf '%s  %s\n' e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2 /home/xander/flash-gordon-live/20260926-abl2/flash-gordon | sha256sum -c -
printf '%s  %s\n' 35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb /home/xander/fg-ring-pack-abliterated/manifest.fgm | sha256sum -c -
printf '%s  %s\n' de8e1067732baaf9bb49083fbdebc55a991c78619d4156624f6ec03ddfc2022b /home/xander/flash-gordon-live/20261015-q4-ablation/flash-gordon | sha256sum -c -
printf '%s  %s\n' __SHA__ '__MANIFEST__' | sha256sum -c -
'@
    Fleet ($check.Replace('__SHA__',$ManifestSHA).Replace('__MANIFEST__',$Manifest)) identity|Out-Null
    Instance stop $original
    Instance stop $Manifest
    Instance start $Manifest
    Ready
    $resident=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'q4-qualify-resident.sh')).
        Replace('manifest=${1:?manifest}',"manifest='$Manifest'")
    $snapshots=Fleet $resident resident
    foreach($snapshot in $snapshots){
        $data=$snapshot.Stdout|ConvertFrom-Json
        if(@($data.processes).Count -ne 1 -or $data.processes[0].exe -ne "$release/flash-gordon"){
            throw 'Candidate process identity mismatch'
        }
    }
    $before=Get-Date
    & pwsh -NoProfile -File "$mainRoot\tools\hillclimb.ps1" -BaseUrl http://192.168.42.42:8080/v1 `
      *> (Join-Path $OutputDir "$Tag-harness.log")
    $code=$LASTEXITCODE
    $file=Get-ChildItem "$mainRoot\results\hillclimb\runs" -Filter '*.json'|
        Where-Object LastWriteTime -ge $before|Sort-Object LastWriteTime -Descending|Select-Object -First 1
    if(-not $file -or $code -notin @(0,2)){throw 'Full canonical harness failed'}
    $runPath=Join-Path $OutputDir "$Tag-harness.json"
    Copy-Item $file.FullName $runPath
    $run=Get-Content $runPath -Raw|ConvertFrom-Json
    if($run.metadata.quick -or $run.metadata.base_url -ne 'http://192.168.42.42:8080/v1'){
        throw 'Wrong harness run selected'
    }
    $gates=Join-Path $OutputDir "$Tag-gates.json"
    if(Test-Path $gates){throw 'Refuse stale harness gate output'}
    python (Join-Path $PSScriptRoot 'q4_qualify.py') --baseline "$mainRoot\results\hillclimb\BASELINE.json" `
      --run $runPath --quality $Quality --output $gates
    if($LASTEXITCODE -notin @(0,1) -or -not (Test-Path $gates)){throw 'Harness gate computation failed'}
    $result=Get-Content $gates -Raw|ConvertFrom-Json
    if($result.passed -and $PromoteOnPass){Ready;$endStateReady=$true;$promoted=$true}
}finally{
    try{
        if(-not $promoted){
            $stopped=$false
            try{Instance stop $Manifest;$stopped=$true}
            finally{
                try{
                    if($stopped){
                        $pack=$Manifest.Substring(0,$Manifest.LastIndexOf('/'))
                        Fleet "set -e;find '$pack' -maxdepth 1 -type f -name 'qsa-*.state*' -delete;df -B1 /home/xander" cleanup|Out-Null
                    }
                }finally{
                    Instance start $original $originalRelease
                    Ready
                    $endStateReady=$true
                }
            }
        }
    }finally{
        @{promoted=$promoted;healthy=$endStateReady;manifest=$(if($promoted){$Manifest}else{$original})
          release=$(if($promoted){$release}else{$originalRelease});checked_at=(Get-Date -Format o)}|
            ConvertTo-Json|Set-Content (Join-Path $OutputDir "$Tag-end-state.json")
        Remove-Item $command -ErrorAction SilentlyContinue
    }
}
