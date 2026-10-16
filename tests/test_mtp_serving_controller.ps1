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
try {
    foreach($kind in @('success','evidence','download','restore','greedy','cleanup')){
        Test-Finalizer $kind
    }
} finally {
    if(Test-Path $base){Remove-Item -Recurse -Force $base}
}
