#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PerformanceResult,
    [Parameter(Mandatory)][string]$LeaseFile,
    [Parameter(Mandatory)][string]$Output,
    [switch]$PlanOnly
)
$ErrorActionPreference='Stop'
. "$PSScriptRoot\ab-stack-common.ps1"
. "$PSScriptRoot\hillclimb-ab.ps1"
$result=Get-Content $PerformanceResult -Raw|ConvertFrom-Json -AsHashtable
$decision=Get-ABDecision $result.metrics ([bool]$result.quality_pass)
if(-not $result.complete-or-not $result.eligible-or-not $decision.eligible-or
    -not $result.same_kernel_hash_identity){
    throw 'Only a complete quality-approved, primary-gate and exact-grouping winner may be pinned.'
}
$B=$result.config.B
$A=$result.config.A
Assert-ABDeployment $B
Assert-ABStackLease $LeaseFile
if((Get-FileHash $result.config.quality_file -Algorithm SHA256).Hash.ToLowerInvariant()-ne
    $result.quality_receipt_sha256){throw 'Quality receipt changed after qualification.'}
if(Test-Path $Output){throw 'Pinned-selection output must be fresh.'}
$Output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$null=New-Item -ItemType Directory $Output
$operator='/home/xander/flash-gordon-live/20261015-abq-operator'
$tag=(Get-ABDeploymentIdentity $B).Substring(0,12)
$step=0
function Fleet([string]$Text,[int[]]$Ranks=@(0..7)){
    Assert-ABStackLease $LeaseFile
    $script=Join-Path $Output 'command.sh'
    [IO.File]::WriteAllText($script,$Text.Replace("`r",''))
    $r=@(& "$PSScriptRoot\ab-fleet.ps1" -ScriptPath $script `
        -Targets @($Ranks|ForEach-Object{"192.168.42.$(42+$_)"}) -PassThru -CommandTimeout 180)
    $r|ConvertTo-Json -Depth 12|Set-Content (Join-Path $Output "$script:step.json")
    $script:step++
    if($r.Count-ne$Ranks.Count-or@($r|Where-Object ExitCode -ne 0).Count){throw 'Pinned fleet operation failed.'}
}
function Encoded([string]$Text){[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Text))}
$source=(Get-Content "$PSScriptRoot\ab-instance.sh" -Raw).Replace("`r",'')
$flags=Get-ABStackFlags $B
$assignments=@($flags.Keys|Where-Object{$_-ne'FG_EXPERT_Q8_WIDE'}|
    ForEach-Object{"$_='$($flags[$_])'"})-join' '
$source=$source.Replace('# AB_STACK_FLAGS',"flags+=($assignments)")
$treeCommand=Get-ABShaderTreeCommand $B
$check=@"
if [[ `$action == start ]]; then
printf '%s  %s\n' '$($B.binary_sha256)' '$($B.release)/flash-gordon' | sha256sum -c -
printf '%s  %s\n' '$($B.manifest_sha256)' '$($B.manifest)' | sha256sum -c -
shader_hash=`$($treeCommand)
test "`$shader_hash" = '$($B.shader_tree_sha256)'
fi
"@
$source=$source.Replace('case "$action" in',"$check`ncase `"`$action`" in").Replace("`r",'')
$instance=Join-Path $Output "ab-instance-stack-$tag.sh"
[IO.File]::WriteAllText($instance,$source)
$sourceHash=(Get-FileHash $instance -Algorithm SHA256).Hash.ToLowerInvariant()
$launcher=@"
#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\n' '$sourceHash' '$operator/ab-instance-stack-$tag.sh' | sha256sum -c -
bash '$operator/ab-instance-stack-$tag.sh' "`${1:?start,stop,status}" '$($B.release)' '$($B.manifest)' '$($B.sessions)' '$($B.wide)'
"@
$launcher=$launcher.Replace("`r",'')
[IO.File]::WriteAllText((Join-Path $Output "launch-stack-$tag.sh"),$launcher)
$json=$B|ConvertTo-Json -Depth 25
[IO.File]::WriteAllText((Join-Path $Output 'current.json'),$json)
if($PlanOnly){return}
$pinned=$false
try{
    & "$PSScriptRoot\ab-deploy.ps1" -Deployment $B -Arm selected-stack
    Fleet @"
set -euo pipefail
cd '$operator'
test -f launch-pre-stack-20261016.sh && test ! -L launch-pre-stack-20261016.sh
test -f pre-stack-20261016.json && test ! -L pre-stack-20261016.json
for path in 'ab-instance-stack-$tag.sh' 'launch-stack-$tag.sh' 'stack-$tag.json' launch-current.sh current.json; do
    test ! -L "`$path"
done
printf '%s' '$(Encoded $source)' | base64 -d >'ab-instance-stack-$tag.sh'
printf '%s' '$(Encoded $launcher)' | base64 -d >'launch-stack-$tag.sh'
printf '%s' '$(Encoded $json)' | base64 -d >'stack-$tag.json'
chmod 700 'ab-instance-stack-$tag.sh' 'launch-stack-$tag.sh'
cp 'launch-stack-$tag.sh' launch-current.sh
cp 'stack-$tag.json' current.json
sha256sum launch-current.sh current.json 'ab-instance-stack-$tag.sh'
"@
    Fleet "bash '$operator/launch-current.sh' stop"
    Fleet "bash '$operator/launch-current.sh' start" @(1..7)
    Fleet "bash '$operator/launch-current.sh' start" @(0)
    $ready=$false
    for($i=0;$i-lt180;$i++){
        try{if((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status-eq'ok'){$ready=$true;break}}catch{}
        Start-Sleep -Seconds 2
    }
    if(-not $ready){throw 'Pinned restart API readiness timed out.'}
    $verify=Get-Content "$PSScriptRoot\stack-verify-selected.sh" -Raw
    Fleet ("set -- '$(Encoded $json)'`n"+$verify)
    $greedy=Join-Path $Output 'restart-greedy.json'
    & "$PSScriptRoot\hillclimb.ps1" -GreedyOutput $greedy
    $actual=Get-Content $greedy -Raw|ConvertFrom-Json -AsHashtable
    $reference=Get-Content $result.config.same_kernel_greedy_control -Raw|ConvertFrom-Json -AsHashtable
    if(-not(Test-ABCanonicalHashes $reference.hashes $actual.cases)){throw 'Restart greedy hashes changed.'}
    $health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 10
    $models=Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 10
    if($health.status-ne'ok'-or$models.data[0].id-ne'Qwen3.8-Flash-Next-Uncensored'){throw 'Restart model/health mismatch.'}
    @{health=$health;models=$models;verified=[DateTimeOffset]::Now.ToString('o');
      restart_tested=$true;greedy_identity=$true;deployment=$B;tag=$tag}|
        ConvertTo-Json -Depth 25|Set-Content (Join-Path $Output 'api-health.json')
    $pinned=$true
}finally{
    if(-not $pinned){
        & "$PSScriptRoot\ab-deploy.ps1" -Deployment $A -Arm pin-failure-restore
        Fleet @"
set -euo pipefail
cd '$operator'
cp launch-pre-stack-20261016.sh launch-current.sh
cp pre-stack-20261016.json current.json
"@
    }
}
