#requires -Version 7.0
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$tokens=$null;$parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repo 'tools\Invoke-MTPServingQualification.ps1'),
    [ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw ($parseErrors | Out-String)}
$outer=@($ast.FindAll({
    param($node)
    $node -is [Management.Automation.Language.TryStatementAst] -and
        $node.Finally -and $node.Finally.Extent.Text.Contains('finalizationErrors')
},$true))
if($outer.Count -ne 1){throw 'Expected one qualification finalizer.'}
$finalizer=[scriptblock]::Create($outer[0].Finally.Extent.Text.Trim().Substring(
    1,$outer[0].Finally.Extent.Text.Trim().Length-2))
$base=Join-Path $repo '.build\mtp-controller-finalization-test'
function Check([bool]$condition,[string]$message){
    if(-not $condition){throw $message}
}
function Test-Finalizer([string]$kind){
    $calls=[Collections.Generic.List[string]]::new()
    $all=@('owner0','owner1','owner2','owner3','owner4','owner5','owner6','owner7')
    $run=Join-Path $base $kind
    $null=New-Item -ItemType Directory -Force $run
    $root='unused-mocked-root';$LeaseId="mock-$kind"
    $started=[DateTimeOffset]::UtcNow
    $mutated=$true;$armed=$true;$created=$true;$complete=$false
    $productionVerified=$false;$scratchRemoved=$false;$finalizationErrors=@()
    $failure=if($kind -eq 'success'){$null}else{'original request failure'}
    $before=@{choices=@(@{message=@{content='OK'}})}
    function RestoreProduction {
        $calls.Add('restore')
        if($kind -eq 'restore'){throw 'injected restore failure'}
        return @{choices=@(@{message=@{content=$(if($kind -eq 'greedy'){'DIFFERENT'}else{'OK'})}})}
    }
    function Guard {
        param($Name,$Command,$Targets)
        $calls.Add("$Command-$($Targets[0])")
        if($kind -eq 'evidence' -and $Command -eq 'evidence' -and $Targets[0] -eq 'owner0'){
            throw 'injected evidence failure'
        }
        if($kind -eq 'cleanup' -and $Command -eq 'cleanup'){throw 'injected cleanup failure'}
    }
    function Remote {
        param($Name,$Text,$Targets,$Download,$Local)
        $calls.Add("download-$($Targets[0])")
        if($kind -eq 'download' -and $Targets[0] -eq 'owner0'){
            throw 'injected download failure'
        }
    }
    . $finalizer
    $outcome=Get-Content (Join-Path $run 'outcome.json') -Raw | ConvertFrom-Json
    if($kind -eq 'restore'){
        Check (-not $outcome.productionVerified -and -not $outcome.scratchRemoved) 'Unverified restoration cleaned scratch.'
        Check ($calls.Count -eq 1) 'Unverified restoration proceeded to evidence/cleanup.'
    } else {
        Check $outcome.productionVerified 'Verified restoration not recorded.'
        Check (@($calls | Where-Object {$_ -like 'evidence-*'}).Count -eq 8) 'An evidence failure skipped later owners.'
        Check ($calls[-1] -eq 'cleanup-owner0') 'An evidence/download failure skipped cleanup.'
        Check ($outcome.scratchRemoved -eq ($kind -ne 'cleanup')) 'Cleanup receipt is inaccurate.'
    }
    Check (@($outcome.finalizationErrors).Count -eq $(if($kind -eq 'success'){0}else{1})) 'Finalization errors were lost.'
    if($kind -ne 'success'){Check ($outcome.failure -eq 'original request failure') 'Original request failure was overwritten.'}
    Write-Output "$kind finalization: PASS"
}
$preflightNodes=@($ast.FindAll({
    param($node)
    $node -is [Management.Automation.Language.IfStatementAst] -and
        $node.Extent.Text.StartsWith("if(`$Action -eq 'Run'){") -and
        $node.Extent.Text.Contains('formatAuditSha256')
},$true))
if($preflightNodes.Count -ne 1){throw 'Expected one sealed candidate preflight.'}
$preflightText=$preflightNodes[0].Clauses[0].Item2.Extent.Text.Trim()
$preflight=[scriptblock]::Create($preflightText.Substring(1,$preflightText.Length-2))
function Test-FormatPreflight([string]$kind){
    $Artifacts=Join-Path $base "audit-$kind"
    $null=New-Item -ItemType Directory -Force $Artifacts
    $release=Join-Path $Artifacts 'release.tar.gz'
    [IO.File]::WriteAllText($release,'sealed-test-release')
    $NativeOverlay='mock-native-overlay'
    function Get-FileHash {
        param([string]$Path,[string]$Algorithm)
        if($Path -eq $NativeOverlay){
            return @{Hash='536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b'}
        }
        return @{Hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
            [IO.File]::ReadAllBytes($Path)))}
    }
    $releaseHash=(Get-FileHash $release -Algorithm SHA256).Hash.ToLowerInvariant()
    $audit=@{
        supported=$true;real_pack_gate_passed=$true
        candidate_release_sha256=$releaseHash
        manifest_sha256='0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252'
        native_overlay_sha256='536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b'
    }
    switch($kind){
        'unsupported' {$audit.supported=$false}
        'ungated' {$audit.real_pack_gate_passed=$false}
        'release' {$audit.candidate_release_sha256='different'}
        'manifest' {$audit.manifest_sha256='different'}
        'overlay' {$audit.native_overlay_sha256='different'}
    }
    $auditPath=Join-Path $Artifacts 'format-audit.json'
    $audit | ConvertTo-Json | Set-Content $auditPath
    $prepared=@{
        releaseSha256=$releaseHash
        formatAuditSha256=(Get-FileHash $auditPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if($kind -eq 'missing'){$prepared.Remove('formatAuditSha256')}
    if($kind -eq 'tampered'){Add-Content $auditPath ' '}
    $prepared | ConvertTo-Json | Set-Content (Join-Path $Artifacts 'prepared.json')
    $failure=$null
    try {. $preflight}catch{$failure=$_.Exception.Message}
    if($kind -eq 'success'){
        Check (-not $failure) "Qualified audit rejected: $failure"
    } else {
        Check ($failure -match '(?i)format.*audit|Format/real-Q8') "Invalid audit was not denied before production: $kind / $failure"
    }
    Write-Output "$kind audit preflight: PASS"
}
try {
    $plan=& (Join-Path $repo 'tools\Invoke-MTPServingQualification.ps1') -Action Plan -Profile SingleSessionPerf |
        ConvertFrom-Json
    Check (-not $plan.productionAccess -and $plan.candidateSessions -eq 1) 'Perf profile is not single-session/no-access.'
    Check (($plan.phases | ForEach-Object {$_[0]}) -join ',' -eq 'A0,D3,A1,F1,A2,F2,A3,F3,A4') 'Perf arms are not bracketed.'
    Check (@($plan.smallKernelMasks).Count -eq 1 -and $plan.smallKernelMasks -eq 330) 'Perf mask differs.'
    Check ($plan.phases[1][3] -eq 0 -and $plan.phases[3][3] -eq 1 -and
           $plan.phases[5][3] -eq 2 -and $plan.phases[7][3] -eq 3) 'Fixed/adaptive plan differs.'
    Check ($plan.restore -like '*sessions2 wide1*') 'Perf plan changes restoration.'
    Write-Output 'Single-session adaptive/fixed lease plan: PASS'
    foreach($kind in @('success','evidence','download','restore','greedy','cleanup')){
        Test-Finalizer $kind
    }
    foreach($kind in @('success','missing','tampered','unsupported','ungated','release','manifest','overlay')){
        Test-FormatPreflight $kind
    }
} finally {
    if(Test-Path $base){Remove-Item -Recurse -Force $base}
}
