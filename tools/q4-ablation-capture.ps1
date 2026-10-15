#requires -Version 7.0
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^/home/xander/fg-ring-pack-abliterated-q4(-protected|-teacher|-control|-ablate-[a-z0-9-]+|-mixed)?/manifest\.fgm$')]
    [string]$Manifest,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9-]+$')][string]$Tag,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$Release='/home/xander/flash-gordon-live/20261015-q4-ablation',
    [string]$TeacherDir,
    [string]$Corpus
)
$ErrorActionPreference='Stop'
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$helper=Join-Path $PSScriptRoot 'decode-instance.ps1'
$original='/home/xander/fg-ring-pack-abliterated/manifest.fgm'
function Instance([string]$Action,[string]$Path,[string]$Mode,[int[]]$Ranks=@(0..7),
                  [string]$Binary=$Release){
    & $helper -Action $Action -Manifest $Path -Release $Binary -Mode $Mode `
      -CaptureTag $Tag -Ranks $Ranks -FleetScript $fleet
}
function Stop-Captures {
    foreach($path in @($original,$Manifest)|Select-Object -Unique){Instance stop $path plain}
}
function Clean-State {
    $script=Join-Path $OutputDir "$Tag-state-cleanup.sh"
    $pack=$Manifest.Substring(0,$Manifest.LastIndexOf('/'))
    [IO.File]::WriteAllText($script,"set -e`nfind '$pack' -maxdepth 1 -type f -name 'qsa-*.state*' -printf '%f %s\n' -delete`ndf -B1 /home/xander`n")
    try{
        $r=@(& $fleet -ScriptPath $script -Targets @(42..49|ForEach-Object{"192.168.42.$_"}) -PassThru)
        $r|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputDir "$Tag-state-cleanup.json")
        if($r.Count -ne 8 -or @($r|Where-Object ExitCode -ne 0).Count){throw 'Inactive capture-state cleanup failed'}
    }finally{Remove-Item $script -ErrorAction SilentlyContinue}
}
try {
    Stop-Captures
    Instance start $Manifest quality @(1..7)
    $args=@{BuildDir=$Release;Manifest=$Manifest;OutputDir=(Join-Path $OutputDir $Tag)
            ResidentBatch=$true;CaptureTag=$Tag}
    if($Corpus){$args.Corpus=$Corpus}
    & (Join-Path $PSScriptRoot 'q4-capture.ps1') @args
    if($TeacherDir){
        $qualityPath=Join-Path $OutputDir "$Tag-quality.json"
        if(Test-Path $qualityPath){throw 'Refuse stale quality result'}
        $args=@('--teacher',$TeacherDir,'--candidate',(Join-Path $OutputDir $Tag),
                '--output',$qualityPath)
        if($Corpus){$args+=@('--corpus',$Corpus)}
        & python (Join-Path $PSScriptRoot 'q4_quality.py') @args
        if($LASTEXITCODE -notin @(0,1) -or -not (Test-Path $qualityPath)){
            throw 'Quality computation failed'
        }
    }
} finally {
    $stopped=$false
    try{Stop-Captures;$stopped=$true}
    finally{
        try{if($stopped){Clean-State}}
        finally{
            Instance start $original plain @(0..7) '/home/xander/flash-gordon-live/20260926-abl2'
            $ready=$false
            for($i=0;$i -lt 120;$i++){
                try{
                    $health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2
                    if($health.status -eq 'ok'){$ready=$true;break}
                }catch{}
                Start-Sleep -Seconds 3
            }
            if(-not $ready){throw 'Original API restoration failed'}
            @{release='/home/xander/flash-gordon-live/20260926-abl2';manifest=$original
              restored_at=(Get-Date -Format o);health=$health}|ConvertTo-Json -Depth 4|
                Set-Content (Join-Path $OutputDir "$Tag-restored.json")
        }
    }
}
