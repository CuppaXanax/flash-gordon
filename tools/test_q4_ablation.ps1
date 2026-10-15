#requires -Version 7.0
$ErrorActionPreference='Stop'
$scratch=Join-Path (Get-Location) '.q4-ablation-test'
if(Test-Path $scratch){throw 'Refuse existing ablation test workspace'}
New-Item -ItemType Directory $scratch|Out-Null
function Write-Mock([string]$Name,[string]$Text){
    [IO.File]::WriteAllText((Join-Path $scratch $Name),$Text)
}
function global:Invoke-RestMethod {param($Uri,$TimeoutSec)
    $global:Events+=@{kind='health'}
    if($Uri -like '*/v1/models'){return @{data=@(@{id='mock-model'})}}
    return @{status='ok'}
}
function global:Invoke-WebRequest {param($Uri,$Method,$ContentType,$Body,$TimeoutSec)
    $global:WebCalls++
    $global:CaptureSeen=$true
    $global:Events+=@{kind='request'}
    if($global:WebCalls -eq $global:FailWeb){throw 'Injected HTTP failure'}
    return [pscustomobject]@{Headers=@{};Content='{"usage":{"prompt_tokens":131040}}'}
}
try{
    Copy-Item (Join-Path $PSScriptRoot 'q4-ablation-capture.ps1') $scratch
    Copy-Item (Join-Path $PSScriptRoot 'q4-ablation-profile.ps1') $scratch
    Copy-Item (Join-Path $PSScriptRoot 'q4-ablation-harness.ps1') $scratch
    Write-Mock 'decode-instance.ps1' @'
param($Action,$Manifest,$Release,$Mode,$CaptureTag,$Ranks,$FleetScript)
$global:Events+=@{kind='instance';action=$Action;manifest=$Manifest;release=$Release;mode=$Mode;ranks=@($Ranks)}
if($global:FailStop -and $global:CaptureSeen -and $Action -eq 'stop' -and $Manifest.Contains('-ablate-')){
    throw 'Injected stop failure'
}
'@
    Write-Mock 'q4-capture.ps1' @'
param($BuildDir,$Manifest,$OutputDir,[switch]$ResidentBatch,$CaptureTag,$Corpus)
$global:Events+=@{kind='capture'}
$global:CaptureSeen=$true
if($global:FailCapture){throw 'Injected capture failure'}
'@
    Write-Mock 'Invoke-Q4Fleet.ps1' @'
param($ScriptPath,$Targets,[switch]$PassThru,$DownloadPath,$DownloadSource)
$text=[IO.File]::ReadAllText($ScriptPath)
if($text.Contains("find '/home/xander/fg-ring-pack-abliterated'")){throw 'Original pack targeted for deletion'}
if($Targets.Count -notin @(1,8) -or @($Targets|Where-Object{$_ -notmatch '^192\.168\.42\.4[2-9]$'}).Count){throw 'Unsafe fleet scope'}
$failure=0
if($text.Contains('sha256sum')){
    $global:Events+=@{kind='identity'}
}elseif($text.Contains('resident-snapshot-fixture')){
    $global:Events+=@{kind='resident'}
}elseif($text.Contains('stat -c')){
    $global:Events+=@{kind='offset'}
}elseif($DownloadPath){
    if(-not $text.Contains('tail -c +$((offset+1))')){throw 'Warm profile not excluded'}
    [IO.File]::WriteAllText($DownloadPath,'PREFILL_TEST')
    $global:Events+=@{kind='download'}
}else{
    if(-not $text.Contains("-type f -name 'qsa-*.state*'")){throw 'Unscoped state cleanup'}
    $global:Events+=@{kind='cleanup'}
    $failure=[int]$global:FailCleanup
}
foreach($ip in $Targets){
    $stdout=if($text.Contains('resident-snapshot-fixture')){
        '{"processes":[{"exe":"/home/xander/flash-gordon-live/20261015-q4-ablation/flash-gordon"}]}'
    }else{''}
    [pscustomobject]@{IP=$ip;ExitCode=$failure;Stdout=$stdout;Stderr=''}
}
'@
    $candidate='/home/xander/fg-ring-pack-abliterated-q4-ablate-test/manifest.fgm'
    $original='/home/xander/fg-ring-pack-abliterated/manifest.fgm'
    foreach($case in @('success','capture-failure','cleanup-failure','stop-failure')){
        $global:Events=@()
        $global:CaptureSeen=$false
        $global:FailCapture=$case -eq 'capture-failure'
        $global:FailCleanup=$case -eq 'cleanup-failure'
        $global:FailStop=$case -eq 'stop-failure'
        $failed=$false
        try{
            & (Join-Path $scratch 'q4-ablation-capture.ps1') -Manifest $candidate `
              -Tag $case -OutputDir (Join-Path $scratch $case)
        }catch{$failed=$true}
        if($failed -ne ($case -ne 'success')){throw "Unexpected failure handling: $case"}
        $starts=@($global:Events|Where-Object{
            $_.kind -eq 'instance' -and $_.action -eq 'start' -and $_.manifest -eq $original})
        if($starts.Count -ne 1 -or $starts[0].ranks.Count -ne 8 -or
           $starts[0].release -ne '/home/xander/flash-gordon-live/20260926-abl2'){
            throw "Original restoration missing: $case"
        }
        $cleanups=if($case -eq 'stop-failure'){0}else{1}
        if(@($global:Events|Where-Object kind -eq 'cleanup').Count -ne $cleanups -or
           $global:Events[-1].kind -ne 'health'){throw "Cleanup/readiness ordering: $case"}
    }
    $global:Events=@();$rejected=$false
    try{
        & (Join-Path $scratch 'q4-ablation-capture.ps1') -Manifest $original `
          -Tag invalid -OutputDir (Join-Path $scratch 'invalid')
    }catch{$rejected=$true}
    if(-not $rejected -or $global:Events.Count){throw 'Original capture did not fail closed'}
    foreach($failureAt in @(0,1,2,3)){
        $global:Events=@();$global:CaptureSeen=$false
        $global:FailStop=$false;$global:FailCleanup=$false
        $global:FailWeb=$failureAt;$global:WebCalls=0;$failed=$false
        $wallOnly=$failureAt -eq 3
        try{
            & (Join-Path $scratch 'q4-ablation-profile.ps1') -Manifest $candidate `
              -Tag "profile-$failureAt" -OutputDir (Join-Path $scratch "profile-$failureAt") -WallOnly:$wallOnly
        }catch{$failed=$true}
        if($failed -ne ($failureAt -in @(1,2))){throw "Profile failure handling: $failureAt"}
        $starts=@($global:Events|Where-Object{
            $_.kind -eq 'instance' -and $_.action -eq 'start' -and $_.manifest -eq $original})
        if($starts.Count -ne 1 -or $starts[0].release -ne '/home/xander/flash-gordon-live/20260926-abl2'){
            throw "Profile original restoration: $failureAt"
        }
        if($global:Events[-1].kind -ne 'health' -or
           @($global:Events|Where-Object kind -eq 'cleanup').Count -ne 8){
            throw "Profile state cleanup/readiness: $failureAt"
        }
        $offsets=if($failureAt -eq 1 -or $wallOnly){0}else{8}
        if(@($global:Events|Where-Object kind -eq 'offset').Count -ne $offsets){
            throw 'Long profile boundary capture missing'
        }
        $start=@($global:Events|Where-Object{
            $_.kind -eq 'instance' -and $_.action -eq 'start' -and $_.manifest -eq $candidate})[0]
        if($start.mode -ne $(if($wallOnly){'plain'}else{'prefill'})){throw 'Wrong profile flag mode'}
        if($wallOnly){
            $a=Get-Content (Join-Path $scratch 'profile-0\profile-0-131072.json') -Raw|ConvertFrom-Json
            $b=Get-Content (Join-Path $scratch 'profile-3\profile-3-131072.json') -Raw|ConvertFrom-Json
            if($b.kernel_profile -or $a.request_sha256 -ne $b.request_sha256){
                throw 'Wall-only request differs from instrumented control'
            }
        }
    }
    foreach($record in @(
        @{passed=$false;samples=480;full_vocab=248320},
        @{passed=$true;samples=80;full_vocab=248320},
        @{passed=$true;samples=480;full_vocab=1024})){
        $global:Events=@();$rejected=$false
        $quality=Join-Path $scratch 'invalid-quality.json'
        $record|ConvertTo-Json|Set-Content $quality
        try{
            & (Join-Path $scratch 'q4-ablation-harness.ps1') `
              -Manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-test/manifest.fgm `
              -ManifestSHA ('a'*64) -Quality $quality -Tag invalid `
              -OutputDir (Join-Path $scratch 'denied-harness')
        }catch{
            if($_.Exception.Message -notlike 'Require a passing*'){throw}
            $rejected=$true
        }
        if(-not $rejected -or $global:Events.Count){throw 'Unqualified harness run did not fail closed'}
    }
    Write-Mock 'q4-qualify-resident.sh' 'manifest=${1:?manifest} # resident-snapshot-fixture'
    function global:pwsh {
        $global:CaptureSeen=$true
        $global:Events+=@{kind='harness'}
        if($global:FailHarness){throw 'Injected canonical harness failure'}
        @{metadata=@{quick=$false;base_url='http://192.168.42.42:8080/v1'}}|
            ConvertTo-Json|Set-Content $global:MockRun
        $global:LASTEXITCODE=0
    }
    function global:python {
        $output=$args[[array]::IndexOf($args,'--output')+1]
        @{passed=$global:GatePassed}|ConvertTo-Json|Set-Content $output
        $global:LASTEXITCODE=if($global:GatePassed){0}else{1}
    }
    function global:Get-ChildItem {
        param($Path,$Filter)
        if($Path -eq 'D:\looking-glass-labs\flash-gordon\results\hillclimb\runs'){
            Get-Item $global:MockRun
        }else{Microsoft.PowerShell.Management\Get-ChildItem -Path $Path -Filter $Filter}
    }
    $quality=Join-Path $scratch 'valid-quality.json'
    @{passed=$true;samples=480;full_vocab=248320}|ConvertTo-Json|Set-Content $quality
    $global:Events=@();$rejected=$false
    try{
        & (Join-Path $scratch 'q4-ablation-harness.ps1') `
          -Manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-aligned-control/manifest.fgm `
          -ManifestSHA ('a'*64) -Quality $quality -Tag control-promote `
          -OutputDir (Join-Path $scratch 'control-promote') -PromoteOnPass
    }catch{
        if($_.Exception.Message -ne 'Q8 performance control is diagnostic only'){throw}
        $rejected=$true
    }
    if(-not $rejected -or $global:Events.Count){throw 'Q8 control promotion did not fail closed'}
    foreach($case in @('reject','promote','harness-failure','cleanup-failure','stop-failure')){
        $global:Events=@();$global:CaptureSeen=$false
        $global:GatePassed=$case -eq 'promote'
        $global:FailHarness=$case -eq 'harness-failure'
        $global:FailCleanup=$case -eq 'cleanup-failure'
        $global:FailStop=$case -eq 'stop-failure'
        $global:MockRun=Join-Path $scratch "run-$case.json"
        $output=Join-Path $scratch "harness-$case"
        $failed=$false
        try{
            & (Join-Path $scratch 'q4-ablation-harness.ps1') `
              -Manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-test/manifest.fgm `
              -ManifestSHA ('a'*64) -Quality $quality -Tag $case -OutputDir $output -PromoteOnPass
        }catch{$failed=$true}
        if($failed -ne ($case -like '*failure')){throw "Harness failure handling: $case"}
        $starts=@($global:Events|Where-Object{
            $_.kind -eq 'instance' -and $_.action -eq 'start' -and $_.manifest -eq $original})
        if($starts.Count -ne [int]($case -ne 'promote')){throw "Harness restore decision: $case"}
        $cleanups=if($case -in @('promote','stop-failure')){0}else{1}
        if(@($global:Events|Where-Object kind -eq 'cleanup').Count -ne $cleanups){
            throw "Harness stopped-state cleanup: $case"
        }
        $state=Get-Content (Join-Path $output "$case-end-state.json") -Raw|ConvertFrom-Json
        if($state.promoted -ne ($case -eq 'promote')){throw "Harness promotion receipt: $case"}
    }
    'Capture/profile/harness rollback, conditional promotion, warm-log exclusion, quality guard: PASS (mock only)'
}finally{
    Remove-Item Function:\Invoke-RestMethod -ErrorAction SilentlyContinue
    Remove-Item Function:\Invoke-WebRequest -ErrorAction SilentlyContinue
    Remove-Item Function:\pwsh -ErrorAction SilentlyContinue
    Remove-Item Function:\python -ErrorAction SilentlyContinue
    Remove-Item Function:\Get-ChildItem -ErrorAction SilentlyContinue
    Remove-Item $scratch -Recurse -Force
}
