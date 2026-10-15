#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][string]$Corpus,
    [Parameter(Mandatory)][string]$TeacherDir,
    [Parameter(Mandatory)][string]$ControlQuality,
    [Parameter(Mandatory)][string]$ControlRestored,
    [Parameter(Mandatory)][string]$FamilyBuild
)
$ErrorActionPreference='Stop'
$deadline=(Get-Date).AddHours(1)
while(-not ((Test-Path $ControlQuality) -and (Test-Path $ControlRestored) -and (Test-Path $FamilyBuild))){
    if((Get-Date) -gt $deadline){throw 'Timed out waiting for control restoration and family build'}
    Start-Sleep -Seconds 30
}
$control=Get-Content $ControlQuality -Raw|ConvertFrom-Json
if($control.samples -ne 480 -or -not $control.passed -or
   $control.mean_kld -gt 1e-7 -or $control.p99_kld -gt 1e-6 -or $control.top1_agreement -ne 1){
    throw 'Aligned Q8 control must be near-zero KL / 100% top-1 before family attribution'
}
$built=@(Get-Content $FamilyBuild -Raw|ConvertFrom-Json)
if($built.Count -ne 8 -or @($built|Where-Object ExitCode -ne 0).Count){throw 'Incomplete family staging'}
$families=@('all','gdn-qkv','gdn-out','gdn-other','qsa','shared-gate','shared-up','shared-down','hc','other')
$summary=@()
foreach($family in $families){
    $seals=@($built|ForEach-Object{
        $m=[regex]::Match($_.Stdout,"FAMILY name=$family sha=([a-f0-9]{64})")
        if(-not $m.Success){throw "Missing owner seal: $family"}
        $m.Groups[1].Value
    }|Sort-Object -Unique)
    if($seals.Count -ne 1){throw "Mismatched owner seals: $family"}
    $tag="fam480-$family"
    $log=Join-Path $OutputDir "$tag-orchestration.log"
    & (Join-Path $PSScriptRoot 'q4-ablation-capture.ps1') `
        -Manifest "/home/xander/fg-ring-pack-abliterated-q4-ablate-$family/manifest.fgm" `
        -Tag $tag -OutputDir $OutputDir -Corpus $Corpus -TeacherDir $TeacherDir *> $log
    $q=Get-Content (Join-Path $OutputDir "$tag-quality.json") -Raw|ConvertFrom-Json
    if($q.samples -ne 480 -or $q.corpus_sha256 -ne $control.corpus_sha256){
        throw 'Mismatched ablation sample cohort'
    }
    $summary+=@{family=$family;seal=$seals[0];samples=$q.samples;mean_kld=$q.mean_kld
                p99_kld=$q.p99_kld;top1_agreement=$q.top1_agreement;passed=$q.passed}
    $summary|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputDir 'family-summary.json')
    Write-Host ("{0}: samples={1} mean={2:G7} p99={3:G7} top1={4:P3} pass={5}" -f
        $family,$q.samples,$q.mean_kld,$q.p99_kld,$q.top1_agreement,$q.passed)
}
