#requires -Version 7.0
$ErrorActionPreference = "Stop"
$stage = Join-Path $PSScriptRoot "q4-stage-pack.ps1"
$directory = Join-Path (Split-Path $PSScriptRoot) ".q4-stage-test"
if (Test-Path -LiteralPath $directory) { throw "Refuse existing test fixture directory" }
$script:calls = [Collections.Generic.List[object]]::new()
$script:testSource = $PSCommandPath
$script:overlaySha = (Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
function Invoke-MockFleet {
    param($ScriptPath, [string[]]$Targets, [switch]$PassThru,
        $UploadPath, $UploadDestination, $DownloadSource, $DownloadPath)
    $text = Get-Content -LiteralPath $ScriptPath -Raw
    foreach ($ip in $Targets) {
        $script:calls.Add([pscustomobject]@{IP=$ip; Text=$text})
        $stdout = "OK"
        if ($DownloadSource) {
            Copy-Item -LiteralPath $script:testSource -Destination $DownloadPath
            $stdout = "OVERLAY_SHA $script:overlaySha"
            if ($script:failure -eq 'overlay') { $stdout = "OVERLAY_SHA $('c' * 64)" }
        } elseif ($text.Contains('BASELINE_SHA ')) {
            $stdout = "BASELINE_SHA $('a' * 64)`n"
            if ($script:failure -eq 'baseline' -and $ip.EndsWith('.48')) {
                $stdout = "BASELINE_SHA $('c' * 64)`n"
            }
            for ($rank=0; $rank -lt 8; $rank++) {
                $address = "192.168.42.$(49-$rank)"
                $stdout += "PLAN_RANK rank=$rank layers=6 resident_weight_bytes=9000 common_bytes=1000 expert_bytes=8000 endpoint=${address}:19000 residency_budget_bytes=12000`n"
            }
        } elseif ($text.Contains('CANDIDATE_SHA ')) {
            $stdout = "CANDIDATE_SHA $('b' * 64)"
            if ($script:failure -eq 'candidate' -and $ip.EndsWith('.48')) {
                $stdout = "CANDIDATE_SHA $('c' * 64)"
            }
        }
        [pscustomobject]@{IP=$ip; ExitCode=0; Stdout=$stdout; Stderr=""}
    }
}
try {
    try {
        & $stage -SourceArchive $PSCommandPath -OutputDir $directory
        throw "Missing lease guard"
    } catch {
        if (-not $_.Exception.Message.Contains('No production access')) { throw }
    }
    if (Test-Path -LiteralPath $directory) { throw "Lease refusal created artifacts" }
    $text = (Get-Content -LiteralPath $stage -Raw).Replace(
        '$fleet = Join-Path $PSScriptRoot "Invoke-Q4Fleet.ps1"',
        '$fleet = "Invoke-MockFleet"')
    $script:failure = ''
    & ([scriptblock]::Create($text)) -SourceArchive $PSCommandPath -OutputDir $directory -Label mock -AfterLease
    $packs = @($script:calls | Where-Object { $_.Text.Contains('CANDIDATE_SHA ') })
    if ($packs.Count -ne 8) { throw "Expected eight owner-local repacks" }
    foreach ($pack in $packs) {
        $rank = 49-[int]$pack.IP.Split('.')[-1]
        if ($pack.Text -notmatch ('pack" '+$rank+' --keep-q8-head --cook-dense')) {
            throw "Manifest-derived rank not used for $($pack.IP)"
        }
        if ($pack.Text.Contains('systemctl') -or $pack.Text.Contains(' worker ')) { throw "Staging deploys a service" }
    }
    if (-not (Test-Path (Join-Path $directory 'staged.json'))) { throw "Missing durable staging evidence" }
    foreach ($failure in @('overlay','baseline','candidate')) {
        Remove-Item -LiteralPath $directory -Recurse -Force
        $script:failure = $failure
        $expected = @{overlay='Overlay transfer SHA256 mismatch'; baseline='Owners disagree'; candidate='Candidate manifest hashes disagree'}
        try {
            & ([scriptblock]::Create($text)) -SourceArchive $PSCommandPath -OutputDir $directory -Label mock -AfterLease
            throw "Failed to reject $failure hash disagreement"
        } catch {
            if (-not $_.Exception.Message.Contains($expected[$failure])) { throw }
        }
    }
    Write-Host "Q4 staging lease guard, transfer hashes, reversed rank map, sealing checks: PASS (mock only)"
} finally {
    if (Test-Path -LiteralPath $directory) { Remove-Item -LiteralPath $directory -Recurse -Force }
}
