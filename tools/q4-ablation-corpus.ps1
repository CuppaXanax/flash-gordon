#requires -Version 7.0
param([Parameter(Mandatory)][string]$OutputDir)
$ErrorActionPreference='Stop'
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir|Out-Null
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$helper=Join-Path $PSScriptRoot 'decode-instance.ps1'
$original='/home/xander/fg-ring-pack-abliterated/manifest.fgm'
$teacher='/home/xander/fg-ring-pack-abliterated-q4-teacher/manifest.fgm'
$release='/home/xander/flash-gordon-live/20261015-q4-ablation'
$command=Join-Path $OutputDir 'corpus-command.sh'
$frames=Join-Path $OutputDir 'teacher-base.fgev'
& python (Join-Path $PSScriptRoot 'q4_make_contexts.py') --base-frames $frames
if($LASTEXITCODE){throw 'Base frame creation failed'}
function Instance([string]$Action,[string]$Manifest,[int[]]$Ranks,[string]$Binary=$release){
    & $helper -Action $Action -Manifest $Manifest -Release $Binary -Mode plain `
      -CaptureTag natural-corpus -Ranks $Ranks -FleetScript $fleet
}
function Fleet([string]$Text,[string[]]$Targets,[hashtable]$Transfers=@{}){
    [IO.File]::WriteAllText($command,$Text.Replace("`r",''))
    $records=@(& $fleet -ScriptPath $command -Targets $Targets -PassThru @Transfers)
    $records|ConvertTo-Json -Depth 6 -Compress|Add-Content (Join-Path $OutputDir 'corpus-fleet.jsonl')
    if(@($records|Where-Object ExitCode -ne 0).Count){throw 'Teacher corpus fleet operation failed'}
}
$prepare=@'
set -euo pipefail
old=/home/xander/fg-ring-pack-abliterated
teacher=/home/xander/fg-ring-pack-abliterated-q4-teacher
if [[ ! -e $teacher ]]; then
  mkdir "$teacher"
  cp --reflink=always "$old/manifest.fgm" "$teacher/manifest.fgm"
  for file in "$old"/*; do
    name=${file##*/}
    case "$name" in manifest.fgm|*.state*) continue ;; esac
    test ! -d "$file" || continue
    ln -s "$file" "$teacher/$name"
  done
  ln -s "$old/tokenizer" "$teacher/tokenizer"
fi
cmp "$old/manifest.fgm" "$teacher/manifest.fgm"
test -s "$teacher/tokenizer/tokenizer.fgt"
'@
Fleet $prepare @(42..49|ForEach-Object{"192.168.42.$_"})
try{
    Instance stop $original @(0..7)
    Instance start $teacher @(1..7)
    $run=@'
set -euo pipefail
build=/home/xander/flash-gordon-live/20261015-q4-ablation
cd "$build"
export FG_SHADER_DIR="$build/vulkan"
export MESA_SHADER_CACHE_DIR="$build/operator/.mesa-cache"
./q4-eval-batch --manifest /home/xander/fg-ring-pack-abliterated-q4-teacher/manifest.fgm --prompts "$build/operator/teacher-base.fgev" --continuations /home/xander/hc-q4/teacher-continuations.fgc --steps 64
'@
    Fleet $run @('192.168.42.42') @{
        UploadPath=$frames;UploadDestination="$release/operator/teacher-base.fgev"
        DownloadSource='/home/xander/hc-q4/teacher-continuations.fgc'
        DownloadPath=(Join-Path $OutputDir 'teacher-continuations.fgc')}
}finally{
    Instance stop $teacher @(0..7)
    Instance start $original @(0..7) '/home/xander/flash-gordon-live/20260926-abl2'
    $ready=$false
    for($i=0;$i -lt 120;$i++){
        try{
            $health=Invoke-RestMethod http://192.168.42.42:8080/health -TimeoutSec 2
            if($health.status -eq 'ok'){$ready=$true;break}
        }catch{}
        Start-Sleep -Seconds 3
    }
    if(-not $ready){throw 'Original API restoration failed after corpus construction'}
    Remove-Item $command -ErrorAction SilentlyContinue
}
& python (Join-Path $PSScriptRoot 'q4_make_contexts.py') `
  --continuations (Join-Path $OutputDir 'teacher-continuations.fgc') `
  --output (Join-Path $OutputDir 'natural480.json')
if($LASTEXITCODE){throw 'Freezing natural teacher frontiers failed'}
