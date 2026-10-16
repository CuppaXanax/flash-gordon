#requires -Version 7.0
param(
    [Parameter(Mandatory)][hashtable]$Deployment,
    [Parameter(Mandatory)][string]$Corpus,
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][string]$LeaseFile
)
$ErrorActionPreference='Stop'
. "$PSScriptRoot\ab-stack-common.ps1"
Assert-ABDeployment $Deployment -Capture
Assert-ABStackLease $LeaseFile
$cleaning=$false
$transportAttempted=$false
$flags=Get-ABStackFlags $Deployment
if($Deployment.capture_program-notmatch'^/home/xander/flash-gordon-live/[0-9]{8}-abq[-a-zA-Z0-9]+/q4-eval-batch$'-or
   $Deployment.capture_sha256-notmatch'^[a-f0-9]{64}$'){throw 'Pin a source-matched staged q4-eval-batch helper.'}
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if(Test-Path $OutputDir){throw 'Refuse stale quality capture.'}
$null=New-Item -ItemType Directory $OutputDir
$owner=[Guid]::NewGuid().ToString('N')
$remote='/home/xander/hc-abq-stack-'+$owner
$script=Join-Path $OutputDir 'command.sh'
function Fleet([string]$Text,[int[]]$Ranks=@(0..7),[hashtable]$Transfers=@{},[switch]$PassRecords){
    if(-not $cleaning){Assert-ABStackLease $LeaseFile}
    [IO.File]::WriteAllText($script,$Text.Replace("`r",''))
    $args=@{ScriptPath=$script;Targets=@($Ranks|ForEach-Object{"192.168.42.$(42+$_)"})
        PassThru=$true;CommandTimeout=7200}
    foreach($key in $Transfers.Keys){$args[$key]=$Transfers[$key]}
    $script:transportAttempted=$true
    $r=@(& "$PSScriptRoot\ab-fleet.ps1" @args)
    $r|ConvertTo-Json -Depth 12 -Compress|Add-Content (Join-Path $OutputDir 'fleet.jsonl')
    if($r.Count-ne$Ranks.Count-or@($r|Where-Object ExitCode -ne 0).Count){throw 'Stack capture fleet step failed.'}
    if($PassRecords){return $r}
}
$frames=Join-Path $OutputDir 'prompts.fgev'
& python "$PSScriptRoot\q4_quality.py" --corpus $Corpus --framed-contexts $frames
if($LASTEXITCODE){throw 'Invalid frozen quality corpus.'}
$prepared=$false
$workersTouched=$false
try{
    $prepared=$true
    $treeCommand=Get-ABShaderTreeCommand $Deployment
    $initial=@(Fleet "set -euo pipefail`ntest ! -e '$remote'`nmkdir -m 700 '$remote'`nprintf '%s' '$owner' >'$remote/owner'`nmkdir '$remote/logits'`nshader_hash=`$($treeCommand)`nprintf 'SHADERS_SHA256=%s\n' `"`$shader_hash`"`n" -PassRecords)
    $hashes=@($initial|ForEach-Object{
        if($_.Stdout-notmatch'(?m)^SHADERS_SHA256=([0-9a-f]{64})\r?$'){throw 'Missing owner shader digest.'}
        $Matches[1]
    }|Sort-Object -Unique)
    if($hashes.Count-ne1-or($Deployment.ContainsKey('shader_tree_sha256')-and
        $Deployment.shader_tree_sha256-ne$hashes[0])){throw 'Owner shader trees disagree with capture identity.'}
    $Deployment.shader_tree_sha256=$hashes[0]
    Assert-ABStackLease $LeaseFile
    $workersTouched=$true
    & "$PSScriptRoot\ab-deploy.ps1" -Deployment $Deployment -Arm quality -CaptureOnly -CaptureDirectory "$remote/logits"
    Fleet 'true' @(0) @{UploadPath=$frames;UploadDestination="$remote/prompts.fgev"}
    $unset=@((Get-ABStackFlagNames)+@('FG_DECODE_PROFILE','FG_DECODE_RING_TRACE','FG_FABRIC_PROFILE',
        'FG_GENERATE_TRACE','FG_DECODE_FUSION','FG_EVAL_LOGITS','FG_EVAL_LOGITS_DIR',
        'FG_DENSE_SOA','FG_DENSE_K_COOKED','FG_RING_TRACE','LD_PRELOAD'))|
        ForEach-Object{"-u $_"}
    $envArgs=@("FG_SHADER_DIR='$($Deployment.release)/vulkan'")
    foreach($key in $flags.Keys){$envArgs+="$key='$($flags[$key])'"}
    $command=@"
set -euo pipefail
    ulimit -c 0
    cd '$remote'
    export TMPDIR='$remote' XDG_CACHE_HOME='$remote/cache'
    printf '%s  %s\n' '$($Deployment.capture_sha256)' '$($Deployment.capture_program)' | sha256sum -c -
printf '%s  %s\n' '$($Deployment.binary_sha256)' '$($Deployment.release)/flash-gordon' | sha256sum -c -
printf '%s  %s\n' '$($Deployment.manifest_sha256)' '$($Deployment.manifest)' | sha256sum -c -
env $($unset -join ' ') $($envArgs -join ' ') '$($Deployment.capture_program)' --manifest '$($Deployment.manifest)' --prompts '$remote/prompts.fgev' >'$remote/coordinator.log' 2>&1 &
pid=`$!
echo "`$pid" >'$remote/capture.pid'
wait "`$pid"
"@
    Fleet $command @(0)
    Fleet "set -euo pipefail`ntar -czf '$remote/logits.tar.gz' -C '$remote/logits' .`n" @(4) @{
        DownloadSource="$remote/logits.tar.gz";DownloadPath=(Join-Path $OutputDir 'logits.tar.gz')}
    $count=@(Get-Content $Corpus -Raw|ConvertFrom-Json).Count
    & python "$PSScriptRoot\ab_stack_archive.py" (Join-Path $OutputDir 'logits.tar.gz') $OutputDir $count
    if($LASTEXITCODE){throw 'Quality extraction failed.'}
    & python "$PSScriptRoot\q4_quality.py" --corpus $Corpus --import-numbered $OutputDir
    if($LASTEXITCODE){throw 'Full-vocabulary quality capture validation failed.'}
    @{deployment=$Deployment;identity=(Get-ABDeploymentIdentity $Deployment);shader_tree_sha256=$Deployment.shader_tree_sha256
      capture_program=$Deployment.capture_program;capture_sha256=$Deployment.capture_sha256}|
        ConvertTo-Json -Depth 20|Set-Content (Join-Path $OutputDir 'capture-identity.json')
}finally{
    $cleaning=$true
    $cleanupErrors=@()
    if($prepared-and$transportAttempted){
        try{Fleet 'true' @(0) @{DownloadSource="$remote/coordinator.log";DownloadPath=(Join-Path $OutputDir 'coordinator.log')}}
        catch{$_|Out-String|Add-Content (Join-Path $OutputDir 'cleanup.log')}
        try{Fleet @"
set -euo pipefail
test -f '$remote/owner' || exit 0
test "`$(cat '$remote/owner')" = '$owner'
if test -f '$remote/capture.pid'; then
  pid=`$(cat '$remote/capture.pid')
  if test -e "/proc/`$pid/exe"; then
    test "`$(readlink "/proc/`$pid/exe")" = '$($Deployment.capture_program)'
    python3 - "`$pid" '$($Deployment.manifest)' '$remote/prompts.fgev' <<'PY'
import sys
from pathlib import Path
a=(Path("/proc")/sys.argv[1]/"cmdline").read_bytes().split(b"\0")
assert any(a[i]==b"--manifest" and a[i+1]==sys.argv[2].encode() for i in range(len(a)-1))
assert any(a[i]==b"--prompts" and a[i+1]==sys.argv[3].encode() for i in range(len(a)-1))
PY
    kill -TERM "`$pid"
    for ((n=0;n<45;n++)); do test -e "/proc/`$pid/exe" || break; sleep 1; done
    test ! -e "/proc/`$pid/exe"
  fi
fi
"@ @(0)}catch{$cleanupErrors+=@($_.Exception.Message)}
        if($workersTouched){
            try{& "$PSScriptRoot\ab-deploy.ps1" -Deployment $Deployment -Arm quality-stop -StopOnly}
            catch{$cleanupErrors+=@($_.Exception.Message)}
        }
        if(-not $cleanupErrors.Count){
            try{Fleet "set -euo pipefail`ntest -f '$remote/owner' || exit 0`ntest `"`$(cat '$remote/owner')`" = '$owner'`ntest -O '$remote' && test ! -L '$remote'`ntest `"`$(readlink -f '$remote')`" = '$remote'`nrm -rf -- '$remote'`n"}
            catch{$cleanupErrors+=@($_.Exception.Message)}
        }
        if($cleanupErrors.Count){throw ($cleanupErrors-join'; ')}
    }
}
