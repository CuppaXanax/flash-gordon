#requires -Version 7.0
<#
.SYNOPSIS
Escalating-context stability gate for the Flash Gordon ring fleet.

.DESCRIPTION
Soak the live OpenAI endpoint with growing contexts, a multi-turn growing
conversation, correctness probes, per-blade liveness checks, the live startup
ledger and the perf band. This is the regression gate for the QSA selection
boundary that killed rank 1 with "invalid causal QSA score tile" during long
ring prefills.

The script never restarts the fleet: it attaches to whatever is serving
http://192.0.2.42:8080.  Run it after every fleet deploy before handing the
endpoint back to a user.  It asserts the serving process's own FG_LEDGER line
(as echoed in the X-Flash-Gordon-Ledger response header) against the sealed
fleet geometry, so a deploy that changes ownership or weight bytes cannot pass
silently.

.OUTPUT FORMAT (one line per record, stable prefixes for grep):
  FGSTAB stage=<name> target=<tokens> prompt=<actual> status=PASS|FAIL content=<chars> prefill_tps=<f> decode_tps=<f> note=<text>
  FGSTAB turn=<n> prompt=<actual> status=PASS|FAIL content=<chars> prefill_tps=<f> note=<text>
  FGSTAB correctness=<name> answer=[...] status=PASS|FAIL
  FGSTAB rank=<n> alive=<0|1> status=PASS|FAIL
  FGSTAB ledger status=PASS|FAIL blocks=<runs> wire_hops=<n> weights_total=<bytes> logical=<tokens> qsa_cache_bytes=<bytes> note=<text>
  FGSTAB perf=<name> value=<f> band=<f> status=PASS|FAIL
  FGSTAB summary stages=<n> conversation=<n> correctness=<n> ranks=<n> status=PASS|FAIL failures=<n>
Exit code 0 only when every line says PASS.

.EXAMPLE
pwsh -NoProfile -File tools/pi-stability.ps1
.EXAMPLE
pwsh -NoProfile -File tools/pi-stability.ps1 -SkipPerf
#>
[CmdletBinding()]
param(
    [string]$ApiUrl = $(if ($env:FG_API_URL) { $env:FG_API_URL } else { "http://192.0.2.42:8080/v1/chat/completions" }),
    [string]$Model = "Qwen3.8-Flash-Next",
    [string]$FleetRunner = $(if ($env:FG_FLEET_RUNNER) { $env:FG_FLEET_RUNNER } else { "D:\workspace\bc-250-dbg\Invoke-BC250Fleet.ps1" }),
    [string]$FleetUser = "xander",
    [int]$RequestTimeoutSec = 1800,
    [double]$MinPrefillTps = 240.0,
    [double]$MinDecodeTps = 22.0,
    # Production expert-parallel layout: rank:layer-count runs in layer order.
    # Override only for a deliberately different placement.
    [string]$ExpectedBlocks = "1:6,0:6,2:6,3:6,4:6,5:6,6:6,7:6",
    # Optional exact per-rank sealed weight bytes, comma list of eight values.
    [string]$ExpectedWeights = "",
    [int]$ExpectedLogical = 262144,
    [switch]$SkipPerf,
    [switch]$SkipRanks,
    [switch]$SkipLedger,
    # TODO(WS1): the 32K QSA page-miss cliff fix is not merged yet
    # (PERFORMANCE_BYTE_BUDGET_2026-09-15.md section 4). The probe is opt-in and
    # reports INFO only. Once WS1 lands, enable the probe by default and set
    # -Min32kDecodeTps to the validated 32K floor (~19-21 TPS today).
    [switch]$Probe32kDecode,
    [double]$Min32kDecodeTps = 0.0,
    [string]$LogPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$script:failures = 0
$script:logged = [Collections.Generic.List[string]]::new()

function Write-Record {
    param([string]$Line)
    $script:logged.Add($Line)
    Write-Host $Line
}

function Add-Failure {
    param([string]$Reason)
    $script:failures++
    Write-Host "FGSTAB failure: $Reason" -ForegroundColor Red
}

function New-Body {
    param([object[]]$Messages, [int]$MaxTokens)
    @{
        model = $Model
        messages = $Messages
        temperature = 0.0
        top_p = 1.0
        top_k = 1
        max_tokens = $MaxTokens
        stream = $false
    } | ConvertTo-Json -Depth 20 -Compress
}

function Invoke-Chat {
    param([object[]]$Messages, [int]$MaxTokens, [string]$Name)
    $body = New-Body -Messages $Messages -MaxTokens $MaxTokens
    try {
        $response = Invoke-WebRequest -Uri $ApiUrl -Method Post `
            -ContentType "application/json" -Body $body `
            -TimeoutSec $RequestTimeoutSec -SkipHttpErrorCheck
    } catch {
        return [pscustomobject]@{ Name = $Name; Ok = $false; Status = 0; Content = ""
            PromptTokens = 0; PrefillTps = 0.0; DecodeTps = 0.0
            Ledger = ""
            Note = "transport: $($_.Exception.Message)" }
    }
    $status = [int]$response.StatusCode
    $content = ""
    $note = ""
    if ($status -eq 200) {
        try {
            $payload = $response.Content | ConvertFrom-Json
            $content = [string]$payload.choices[0].message.content
        } catch {
            $note = "unparseable response: $($_.Exception.Message)"
        }
    } else {
        $note = "http $status"
    }
    function HeaderDouble([string]$headerName) {
        $value = [string]$response.Headers[$headerName]
        if (-not $value) { return 0.0 }
        [double]::Parse($value, $Invariant)
    }
    function HeaderInt([string]$headerName) {
        $value = [string]$response.Headers[$headerName]
        if (-not $value) { return 0 }
        [int]::Parse($value, $Invariant)
    }
    function HeaderText([string]$headerName) {
        return [string]$response.Headers[$headerName]
    }
    [pscustomobject]@{
        Name = $Name
        Ok = $status -eq 200
        Status = $status
        Content = $content
        PromptTokens = HeaderInt "X-Flash-Gordon-Prompt-Tokens"
        PrefillTps = HeaderDouble "X-Flash-Gordon-Prefill-TPS"
        DecodeTps = HeaderDouble "X-Flash-Gordon-Decode-TPS"
        Ledger = HeaderText "X-Flash-Gordon-Ledger"
        Note = $note
    }
}

function Test-Ranks {
    if ($SkipRanks) {
        Write-Record "FGSTAB ranks=skipped"
        return
    }
    if (-not (Test-Path -LiteralPath $FleetRunner)) {
        Add-Failure "fleet runner not found: $FleetRunner"
        Write-Record "FGSTAB ranks=unavailable status=FAIL"
        return
    }
    $scriptText = @'
#!/usr/bin/env bash
ip=$(hostname -I | awk '{print $1}')
rank=$(( ${ip##*.} - 42 ))
count=$(pgrep -cf 'flash-gordon[^ ]* (api|rank|eval)( |$)' || true)
echo "rank=$rank alive=$count"
'@
    $path = Join-Path ([IO.Path]::GetTempPath()) ("fg-stab-{0}.sh" -f [guid]::NewGuid().ToString("N").Substring(0, 8))
    [IO.File]::WriteAllText($path, $scriptText.Replace("`r", ""), [Text.UTF8Encoding]::new($false))
    try {
        $prefix = if ($env:FG_FLEET_PREFIX) { $env:FG_FLEET_PREFIX } else { "192.0.2." }
        $targets = @(42..49 | ForEach-Object { "$prefix$_" })
        try {
            $records = @(& $FleetRunner -ScriptPath $path -Targets $targets -User $FleetUser -Auth Password -PassThru -CommandTimeout 120)
        } catch {
            Add-Failure "fleet runner failed: $($_.Exception.Message)"
            Write-Record "FGSTAB ranks=failed status=FAIL"
            return
        }
        foreach ($ip in $targets) {
            $record = @($records | Where-Object { $_.IP -eq $ip })[0]
            $rank = [int](($ip -split '\.')[-1]) - 42
            if (-not $record) {
                Add-Failure "rank $rank ($ip) returned no fleet record"
                Write-Record "FGSTAB rank=$rank alive=0 status=FAIL"
                continue
            }
            $alive = if ([string]$record.Stdout -match 'alive=(\d+)') { [int]$Matches[1] } else { 0 }
            $status = if ($alive -ge 1 -and $record.ExitCode -eq 0) { "PASS" } else { "FAIL" }
            if ($status -eq "FAIL") {
                Add-Failure "rank $rank ($ip) is not alive: $($record.Stdout) $($record.Stderr)"
            }
            Write-Record "FGSTAB rank=$rank alive=$alive status=$status"
        }
    } finally {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

function New-SoakContent {
    param([int]$TargetTokens)
    # -19 lands the observed prompt counts on 3 mod 4, the residue that
    # triggered the "invalid causal QSA score tile" boundary death.
    $hello = [Math]::Max(90, $TargetTokens - 19)
    "/no_think " + ("hello " * $hello) + " Reply with one word."
}

function Test-Ledger {
    param([object]$Result)
    if ($SkipLedger) {
        Write-Record "FGSTAB ledger=skipped"
        return
    }
    if (-not $Result -or -not $Result.Ledger) {
        Add-Failure "live ledger header is missing; the serving build predates the startup ledger"
        Write-Record "FGSTAB ledger status=FAIL note=missing X-Flash-Gordon-Ledger header"
        return
    }
    $line = [string]$Result.Ledger
    $fields = @{}
    foreach ($token in ($line -split ' ')) {
        $split = $token.IndexOf('=')
        if ($split -gt 0) { $fields[$token.Substring(0, $split)] = $token.Substring($split + 1) }
    }
    $problems = [Collections.Generic.List[string]]::new()
    if (-not $line.StartsWith("FG_LEDGER ")) { $problems.Add("missing FG_LEDGER prefix") }
    $expected = [ordered]@{
        rank = "0"; ranks = "8"; layers = "48"; experts = "512"; topk = "10";
        hidden = "2560"; layer_mode = "single"; batch = "128"; window = "2";
        prefill_frames = "8"; ring_prefill = "1"; ring_decode = "1"
    }
    foreach ($key in $expected.Keys) {
        if ([string]$fields[$key] -ne $expected[$key]) {
            $problems.Add("$key=$([string]$fields[$key]) expected $($expected[$key])")
        }
    }
    if ([string]$fields["blocks"] -ne $ExpectedBlocks) {
        $problems.Add("blocks=$([string]$fields['blocks']) expected $ExpectedBlocks")
    }
    $owners = @()
    foreach ($run in ($ExpectedBlocks -split ',')) {
        $parts = $run -split ':'
        if ($parts.Count -ne 2) { $owners = @(); break }
        $owners += [int]$parts[0]
    }
    if ($owners.Count -gt 0) {
        $expectedHops = 0
        if ($owners[0] -ne 0) { $expectedHops++ }
        $expectedHops += $owners.Count - 1
        if ($owners[-1] -ne 4) { $expectedHops++ }
        if ([string]$fields["wire_hops"] -ne [string]$expectedHops) {
            $problems.Add("wire_hops=$([string]$fields['wire_hops']) expected $expectedHops")
        }
    }
    $weights = @()
    if ($fields["weights"]) { $weights = @($fields["weights"] -split ',') }
    if ($weights.Count -ne 8) {
        $problems.Add("weights has $($weights.Count) entries, expected 8")
    } else {
        foreach ($value in $weights) {
            if ($value -notmatch '^\d+$' -or [uint64]$value -lt 1GB) {
                $problems.Add("weight bytes '$value' are not a plausible per-rank total")
                break
            }
        }
        if ($ExpectedWeights -and ($weights -join ',') -ne $ExpectedWeights) {
            $problems.Add("weights=$($weights -join ',') expected $ExpectedWeights")
        }
    }
    if ($fields["weights_total"] -notmatch '^\d+$' -or [uint64]$fields["weights_total"] -lt 1GB) {
        $problems.Add("weights_total=$([string]$fields['weights_total'])")
    }
    if ([string]$fields["logical"] -ne [string]$ExpectedLogical) {
        $problems.Add("logical=$([string]$fields['logical']) expected $ExpectedLogical")
    }
    if ($fields["qsa_cache_pages"] -notmatch '^\d+$' -or [int]$fields["qsa_cache_pages"] -lt 1) {
        $problems.Add("qsa_cache_pages=$([string]$fields['qsa_cache_pages'])")
    }
    if ($fields["qsa_cache_bytes"] -notmatch '^\d+$' -or [uint64]$fields["qsa_cache_bytes"] -lt 16MB) {
        $problems.Add("qsa_cache_bytes=$([string]$fields['qsa_cache_bytes']) below the sealed 16 MiB floor")
    }
    $status = if ($problems.Count -eq 0) { "PASS" } else { "FAIL" }
    if ($status -eq "FAIL") {
        Add-Failure "ledger: $($problems -join '; ')"
    }
    Write-Record ("FGSTAB ledger status={0} blocks={1} wire_hops={2} weights_total={3} logical={4} qsa_cache_bytes={5} note={6}" -f
        $status, [string]$fields["blocks"], [string]$fields["wire_hops"],
        [string]$fields["weights_total"], [string]$fields["logical"],
        [string]$fields["qsa_cache_bytes"],
        $(if ($problems.Count -eq 0) { "sealed geometry matches" } else { $problems -join "; " }))
}

$started = Get-Date
Write-Record ("FGSTAB start={0} api={1} min_prefill_tps={2} min_decode_tps={3}" -f
    $started.ToString("yyyy-MM-ddTHH:mm:ss"), $ApiUrl, $MinPrefillTps, $MinDecodeTps)

$soak = @(128, 1024, 4096, 8192, 12288, 16384)
$soakResults = @{}
foreach ($target in $soak) {
    $messages = @(@{ role = "user"; content = (New-SoakContent -TargetTokens $target) })
    $result = Invoke-Chat -Messages $messages -MaxTokens 1 -Name "soak-$target"
    $contentChars = $result.Content.Trim().Length
    $lowerBound = [int]($target * 0.9)
    $status = "PASS"
    if (-not $result.Ok) { $status = "FAIL" }
    elseif ($result.PromptTokens -lt $lowerBound) { $status = "FAIL" }
    elseif ($contentChars -le 0) { $status = "FAIL" }
    if ($status -eq "FAIL") {
        Add-Failure "soak ${target}: status=$($result.Status) prompt=$($result.PromptTokens) content=$contentChars $($result.Note)"
    }
    $soakResults[$target] = $result
    Write-Record ("FGSTAB stage=soak target={0} prompt={1} status={2} content={3} prefill_tps={4:F2} decode_tps={5:F2} note={6}" -f
        $target, $result.PromptTokens, $status, $contentChars, $result.PrefillTps, $result.DecodeTps, $result.Note)
    Test-Ranks
}

if ($soakResults.ContainsKey(128)) {
    Test-Ledger -Result $soakResults[128]
} else {
    Add-Failure "the first soak request did not run; cannot assert the live ledger"
    Write-Record "FGSTAB ledger status=FAIL note=no probe response"
}

$conversation = @()
$turnSizes = @(1024, 2048, 4096, 8192)
$previousPrompt = 0
for ($turn = 0; $turn -lt $turnSizes.Count; $turn++) {
    $messages = @($conversation) + @(@{ role = "user"; content = (New-SoakContent -TargetTokens $turnSizes[$turn]) })
    $result = Invoke-Chat -Messages $messages -MaxTokens 8 -Name "conversation-$($turn + 1)"
    $contentChars = $result.Content.Trim().Length
    $status = "PASS"
    if (-not $result.Ok) { $status = "FAIL" }
    elseif ($contentChars -le 0) { $status = "FAIL" }
    elseif ($result.PromptTokens -le $previousPrompt) { $status = "FAIL" }
    if ($status -eq "FAIL") {
        Add-Failure "conversation turn $($turn + 1): status=$($result.Status) prompt=$($result.PromptTokens) content=$contentChars $($result.Note)"
    }
    $previousPrompt = $result.PromptTokens
    $conversation += @{ role = "user"; content = (New-SoakContent -TargetTokens $turnSizes[$turn]) }
    $conversation += @{ role = "assistant"; content = $result.Content }
    Write-Record ("FGSTAB turn={0} prompt={1} status={2} content={3} prefill_tps={4:F2} note={5}" -f
        ($turn + 1), $result.PromptTokens, $status, $contentChars, $result.PrefillTps, $result.Note)
}
Test-Ranks

$correctness = @(
    @{ Name = "twelve"; Prompt = "/no_think What is seven plus five? Answer with just the number."; Match = "12" },
    @{ Name = "paris"; Prompt = "/no_think What is the capital of France? Answer in one word."; Match = "Paris" }
)
foreach ($probe in $correctness) {
    $result = Invoke-Chat -Messages @(@{ role = "user"; content = $probe.Prompt }) -MaxTokens 64 -Name $probe.Name
    $answer = $result.Content.Trim()
    $status = if ($result.Ok -and $answer -match [regex]::Escape($probe.Match)) { "PASS" } else { "FAIL" }
    if ($status -eq "FAIL") {
        Add-Failure "correctness $($probe.Name): status=$($result.Status) answer=[$answer] $($result.Note)"
    }
    Write-Record "FGSTAB correctness=$($probe.Name) answer=[$answer] status=$status"
}

if (-not $SkipPerf) {
    if (-not $soakResults.ContainsKey(4096)) {
        Add-Failure "4K soak request did not run; cannot assert the prefill band"
    } else {
        $fourK = $soakResults[4096]
        $prefillStatus = if ($fourK.PrefillTps -ge $MinPrefillTps) { "PASS" } else { "FAIL" }
        if ($prefillStatus -eq "FAIL") {
            Add-Failure "4K prefill $($fourK.PrefillTps) below $MinPrefillTps"
        }
        Write-Record ("FGSTAB perf=4k_prefill value={0:F2} band={1:F2} status={2}" -f
            $fourK.PrefillTps, $MinPrefillTps, $prefillStatus)
    }
    $short = Invoke-Chat -Messages @(@{ role = "user"; content = "/no_think Give a comma-separated list of distinct GPU inference terms." }) -MaxTokens 32 -Name "short-decode"
    $decodeStatus = if ($short.Ok -and $short.DecodeTps -ge $MinDecodeTps) { "PASS" } else { "FAIL" }
    if ($decodeStatus -eq "FAIL") {
        Add-Failure "short decode $($short.DecodeTps) below $MinDecodeTps (status=$($short.Status) $($short.Note))"
    }
    Write-Record ("FGSTAB perf=short_decode value={0:F2} band={1:F2} status={2}" -f
        $short.DecodeTps, $MinDecodeTps, $decodeStatus)
    Write-Record ("FGSTAB perf=short_prefill value={0:F2} status=INFO" -f $short.PrefillTps)
    # TODO(WS1): the 32K QSA page-miss cliff fix is not merged yet. This probe
    # is opt-in and INFO-only; once WS1 lands, run it by default and assert
    # -Min32kDecodeTps (validated 32K floor, ~19-21 TPS on the current pack).
    if ($Probe32kDecode) {
        $long = Invoke-Chat -Messages @(@{ role = "user"; content = (New-SoakContent -TargetTokens 32768) }) -MaxTokens 32 -Name "32k-decode"
        if ($Min32kDecodeTps -gt 0.0) {
            $longStatus = if ($long.Ok -and $long.DecodeTps -ge $Min32kDecodeTps) { "PASS" } else { "FAIL" }
            if ($longStatus -eq "FAIL") {
                Add-Failure "32K decode $($long.DecodeTps) below $Min32kDecodeTps (status=$($long.Status) $($long.Note))"
            }
            Write-Record ("FGSTAB perf=32k_decode value={0:F2} band={1:F2} status={2}" -f
                $long.DecodeTps, $Min32kDecodeTps, $longStatus)
        } else {
            Write-Record ("FGSTAB perf=32k_decode value={0:F2} band=TODO-WS1 status=INFO" -f
                $long.DecodeTps)
        }
    }
}

Test-Ranks

$status = if ($script:failures -eq 0) { "PASS" } else { "FAIL" }
$summary = ("FGSTAB summary stages={0} conversation={1} correctness={2} ranks={3} status={4} failures={5}" -f
    $soak.Count, $turnSizes.Count, $correctness.Count, 8, $status, $script:failures)
Write-Record $summary
if ($LogPath) {
    Set-Content -LiteralPath $LogPath -Value $script:logged -Encoding utf8
    Write-Host "FGSTAB log=$LogPath"
}
exit $(if ($status -eq "PASS") { 0 } else { 1 })
