#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ConfigPath,
    [Parameter(Mandatory)][string]$Output,
    [switch]$Execute,
    [string]$LeaseFile = '',
    [string]$CaptureScript = (Join-Path $PSScriptRoot 'ab-stack-capture.ps1'),
    [string]$PerformanceScript = (Join-Path $PSScriptRoot 'hillclimb.ps1'),
    [string]$QualityScript = (Join-Path $PSScriptRoot 'q4_quality.py'),
    [string]$DeploymentScript = (Join-Path $PSScriptRoot 'ab-deploy.ps1')
)
$ErrorActionPreference='Stop'
. "$PSScriptRoot\ab-stack-common.ps1"
$config=Get-Content $ConfigPath -Raw|ConvertFrom-Json -AsHashtable
$Output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path $Output){throw 'Candidate stack output must be fresh.'}
$rounds=if($config.ContainsKey('rounds')){[int]$config.rounds}else{6}
if($rounds-lt5-or$rounds-gt30){throw 'Predeclare 5-30 pairs; default is six.'}
$corpus=(Resolve-Path $config.corpus).Path
$items=@(Get-Content $corpus -Raw|ConvertFrom-Json)
if($items.Count-lt400-or$items.Count-gt10000){throw 'Use a frozen 400-10000-frontier quality corpus, not the 80-row screening set.'}
Assert-ABDeployment $config.A -Capture
$names=@{}
foreach($candidate in $config.candidates){
    if($candidate.name-notmatch'^[a-z0-9][a-z0-9-]*$'-or$names.ContainsKey($candidate.name)){
        throw 'Candidate names must be unique safe directory names.'
    }
    Assert-ABDeployment $candidate.B -Capture
    $groupFlags=Get-ABStackFlags $candidate.B
    if($groupFlags['FG_DECODE_SUBMIT_GROUP']-in@('1','2')){
        if(-not $candidate.ContainsKey('hash_control')-or-not $names.ContainsKey($candidate.hash_control)){
            throw 'Submission grouping requires an earlier same-kernel hash_control candidate.'
        }
    }
    if($candidate.ContainsKey('hash_control')){
        if(-not $names.ContainsKey($candidate.hash_control)-or
            (Get-ABKernelIdentity $candidate.B)-ne(Get-ABKernelIdentity $names[$candidate.hash_control].B)){
            throw 'hash_control must precede the candidate and differ only in submission grouping.'
        }
    }
    $names[$candidate.name]=$candidate
    foreach($arm in @($config.A,$candidate.B)){
        if($arm.sessions-ne1){throw 'Candidate flag stacks use sessions 1.'}
        if($arm.binary_sha256-notmatch'^[a-f0-9]{64}$'-or$arm.manifest_sha256-notmatch'^[a-f0-9]{64}$'-or
           $arm.capture_sha256-notmatch'^[a-f0-9]{64}$'){
            throw 'Pin serving binary, manifest and source-matched quality helper hashes before planning.'
        }
    }
}
if(-not $names.Count){throw 'No candidate flag stacks.'}
if($Execute){
    if(-not $LeaseFile){throw 'Execution requires -LeaseFile; planning never contacts the fleet.'}
    Assert-ABStackLease $LeaseFile
}
$null=New-Item -ItemType Directory -Force $Output
$plan=[ordered]@{schema=1;mode='candidate-stack';executed=[bool]$Execute;rounds=$rounds
    not_before=@('multi-session lease released','MTP lease released','new exclusive .42-.49 lease')
    config=$config;corpus_file_sha256=(Get-FileHash $corpus -Algorithm SHA256).Hash.ToLowerInvariant()
    results=@();complete=$false}
$path=Join-Path $Output 'stack-result.json'
$plan|ConvertTo-Json -Depth 40|Set-Content $path
if(-not $Execute){Write-Host "Offline stack plan: $path";return}
$restore=Join-Path $Output 'restore-baseline.ps1'
$baseline=Join-Path $Output 'baseline.json'
$config.A|ConvertTo-Json -Depth 20|Set-Content $baseline
$deploy=(Resolve-Path $DeploymentScript).Path.Replace("'","''")
$baselineQuoted=$baseline.Replace("'","''")
[IO.File]::WriteAllText($restore,"`$c=Get-Content '$baselineQuoted' -Raw|ConvertFrom-Json -AsHashtable`n& '$deploy' -Deployment `$c -Arm restore-baseline`n")
$leasedDeploy=Join-Path $Output 'deploy-leased.ps1'
$common=(Join-Path $PSScriptRoot 'ab-stack-common.ps1').Replace("'","''")
$leaseQuoted=(Resolve-Path $LeaseFile).Path.Replace("'","''")
[IO.File]::WriteAllText($leasedDeploy,"param([hashtable]`$Deployment,[string]`$Arm)`n. '$common'`nAssert-ABStackLease '$leaseQuoted'`n& '$deploy' -Deployment `$Deployment -Arm `$Arm`n")
try{
    $teacher=Join-Path $Output 'teacher'
    & $CaptureScript -Deployment $config.A -Corpus $corpus -OutputDir $teacher -LeaseFile $LeaseFile
    if(-not $?){throw 'Teacher capture failed.'}
    $config.A.shader_tree_sha256=(Get-Content (Join-Path $teacher 'capture-identity.json') -Raw|ConvertFrom-Json).shader_tree_sha256
    $config.A|ConvertTo-Json -Depth 20|Set-Content $baseline
    foreach($candidate in $config.candidates){
        Assert-ABStackLease $LeaseFile
        if($candidate.ContainsKey('hash_control')){
            $controlPath=Join-Path $Output "$($candidate.hash_control)\perf\ab-result.json"
            if(-not(Test-Path $controlPath)){
                $plan.results+=@(@{name=$candidate.name;eligible=$false;perf_run=$false
                    blocked='The same-kernel greedy control did not complete quality and performance.'})
                $plan|ConvertTo-Json -Depth 40|Set-Content $path
                continue
            }
        }
        $dir=Join-Path $Output $candidate.name
        $null=New-Item -ItemType Directory $dir
        $capture=Join-Path $dir 'capture'
        & $CaptureScript -Deployment $candidate.B -Corpus $corpus -OutputDir $capture -LeaseFile $LeaseFile
        if(-not $?){throw 'Candidate capture failed.'}
        $candidate.B.shader_tree_sha256=(Get-Content (Join-Path $capture 'capture-identity.json') -Raw|ConvertFrom-Json).shader_tree_sha256
        $quality=Join-Path $dir 'quality.json'
        & python $QualityScript --corpus $corpus --teacher $teacher --candidate $capture --output $quality
        if($LASTEXITCODE-notin@(0,1)-or-not(Test-Path $quality)){throw 'KL computation failed.'}
        $numeric=Get-Content $quality -Raw|ConvertFrom-Json -AsHashtable
        if($numeric.passed-isnot[bool]-or$numeric.passed-ne($LASTEXITCODE-eq0)){
            throw 'Quality status disagrees with comparator exit status.'
        }
        $row=[ordered]@{name=$candidate.name;quality_pass=[bool]$numeric.passed;eligible=$false;perf_run=$false}
        if($numeric.passed){
            $receipt=Join-Path $dir 'quality-receipt.json'
            @{manifest_sha256=$candidate.B.manifest_sha256;quality_path=$quality
              quality_sha256=(Get-FileHash $quality -Algorithm SHA256).Hash.ToLowerInvariant()
              A_identity=(Get-ABDeploymentIdentity $config.A);B_identity=(Get-ABDeploymentIdentity $candidate.B)
              corpus_sha256=$numeric.corpus_sha256}|ConvertTo-Json -Depth 10|Set-Content $receipt
            $ab=Join-Path $dir 'ab-config.json'
            $abDesign=@{A=$config.A;B=$candidate.B;change_kind='numeric';quality_file=$receipt
              deploy_script=$leasedDeploy;restore_script=$restore
              contexts=@(4096,32768)}
            if($candidate.ContainsKey('hash_control')){
                $abDesign.exact_hash_reference=$controlPath
            }
            $abDesign|ConvertTo-Json -Depth 30|Set-Content $ab
            Assert-ABStackLease $LeaseFile
            & $PerformanceScript -ABConfig $ab -ABRounds $rounds -ABOrder ABBA -ABOutput (Join-Path $dir 'perf')
            $perf=Get-Content (Join-Path $dir 'perf\ab-result.json') -Raw|ConvertFrom-Json
            if(-not $perf.complete){throw 'Incomplete stack performance experiment.'}
            $row.perf_run=$true;$row.eligible=[bool]$perf.eligible
            if($candidate.ContainsKey('hash_control')){
                $row.hash_control=$candidate.hash_control
                $row.same_kernel_hash_identity=[bool]$perf.same_kernel_hash_identity
                $row.eligible=$row.eligible-and$row.same_kernel_hash_identity
            }
        }
        $plan.results+=@($row)
        $plan|ConvertTo-Json -Depth 40|Set-Content $path
    }
    $plan.complete=$true
}catch{$plan.error=$_.Exception.Message;throw}
finally{
    try{
        & $restore
        if(-not $?){throw 'Baseline restore failed.'}
        $plan.restored_baseline=$true
    }catch{$plan.restore_error=$_.Exception.Message;throw}
    finally{$plan|ConvertTo-Json -Depth 40|Set-Content $path}
}
