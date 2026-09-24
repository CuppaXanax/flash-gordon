#requires -Version 7.0
<#
Concurrent-probe acceptance evidence: same generation with and without 4
parallel /health+models probe streams, comparing engine prefill/decode TPS.

FGCC lines are the citable output.
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:FG_BASE_URL) { $env:FG_BASE_URL } else { "http://192.168.42.42:8080" }),
    [int]$PromptTokens = 8000,
    [int]$MaxTokens = 128,
    [int]$ProbeWorkers = 4,
    [int]$ProbeIntervalMs = 50,
    [string]$OutFile = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$log = [Collections.Generic.List[string]]::new()
function Emit([string]$Line) { $log.Add($Line); Write-Host $Line }

function New-Client([int]$TimeoutSec) {
    $handler = [System.Net.Http.SocketsHttpHandler]::new()
    $handler.MaxConnectionsPerServer = 16
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    return $client
}
$client = New-Client 3600

$builder = [Text.StringBuilder]::new()
$sentence = "The architecture of the system is described in this section. "
while (($builder.Length / 4) -lt $PromptTokens) { [void]$builder.Append($sentence) }
$prompt = $builder.ToString()
$body = @{ model = "Qwen3.8-Flash-Next"
    messages = @(@{ role = "user"; content = $prompt })
    temperature = 0.0; top_p = 1.0; top_k = 1
    max_tokens = $MaxTokens; stream = $false } | ConvertTo-Json -Depth 10 -Compress

function Run-Generation([bool]$WithProbes, [int]$RunIndex) {
    $probeDir = Join-Path ([IO.Path]::GetTempPath()) ("fgcc-probes-{0}-{1}" -f $RunIndex, [guid]::NewGuid().ToString("N").Substring(0, 6))
    $null = New-Item -ItemType Directory -Path $probeDir -Force
    $stopFile = Join-Path $probeDir "stop"
    $jobs = @()
    if ($WithProbes) {
        for ($w = 0; $w -lt $ProbeWorkers; $w++) {
            $jobFile = Join-Path $probeDir "worker-$w.txt"
            $jobs += Start-ThreadJob -ScriptBlock {
                param($Base, $Stop, $Out, $Interval)
                $c = [System.Net.Http.HttpClient]::new()
                $c.Timeout = [TimeSpan]::FromSeconds(5)
                while (-not (Test-Path -LiteralPath $Stop)) {
                    $sw = [Diagnostics.Stopwatch]::StartNew()
                    try {
                        $r = $c.GetAsync("$Base/health").GetAwaiter().GetResult()
                        $null = $r.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                    } catch { }
                    $sw.Stop()
                    Add-Content -LiteralPath $Out -Value ("{0:F1}" -f $sw.Elapsed.TotalMilliseconds)
                    Start-Sleep -Milliseconds $Interval
                }
            } -ArgumentList $BaseUrl, $stopFile, $jobFile, $ProbeIntervalMs
        }
    }
    $content = [System.Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8,
        "application/json")
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $resp = $client.PostAsync("$BaseUrl/v1/chat/completions", $content).GetAwaiter().GetResult()
    $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $sw.Stop()
    if ($WithProbes) {
        $null = New-Item -ItemType File -Path $stopFile -Force
        $jobs | Wait-Job | Out-Null
        $jobs | Remove-Job -Force
    }
    $prefillTps = 0.0; $decodeTps = 0.0; $promptActual = 0; $completion = 0
    if ($resp.StatusCode -eq 200) {
        if ($resp.Headers.Contains("X-Flash-Gordon-Prefill-TPS")) {
            $prefillTps = [double]($resp.Headers.GetValues("X-Flash-Gordon-Prefill-TPS")[0])
        }
        if ($resp.Headers.Contains("X-Flash-Gordon-Decode-TPS")) {
            $decodeTps = [double]($resp.Headers.GetValues("X-Flash-Gordon-Decode-TPS")[0])
        }
        $json = $text | ConvertFrom-Json
        $promptActual = [int]$json.usage.prompt_tokens
        $completion = [int]$json.usage.completion_tokens
    }
    $lat = @()
    if ($WithProbes) {
        foreach ($f in Get-ChildItem -Path $probeDir -Filter "worker-*.txt") {
            $lat += Get-Content -LiteralPath $f.FullName | ForEach-Object { [double]$_ }
        }
    }
    Remove-Item -Recurse -Force -LiteralPath $probeDir -ErrorAction SilentlyContinue
    return [pscustomobject]@{ PrefillTps = $prefillTps; DecodeTps = $decodeTps
        Prompt = $promptActual; Completion = $completion; Wall = $sw.Elapsed.TotalSeconds
        Probes = $lat }
}

$controls = @()
for ($i = 0; $i -lt 2; $i++) {
    $r = Run-Generation $false (100 + $i)
    $controls += $r
    Emit ("FGCC control run={0} prompt={1} completion={2} prefill_tps={3:F2} decode_tps={4:F2} wall_s={5:F1}" -f
        $i, $r.Prompt, $r.Completion, $r.PrefillTps, $r.DecodeTps, $r.Wall)
}
$probes = @()
for ($i = 0; $i -lt 2; $i++) {
    $r = Run-Generation $true (200 + $i)
    $probes += $r
    $max = 0.0; $avg = 0.0
    if ($r.Probes.Count) {
        $max = ($r.Probes | Measure-Object -Maximum).Maximum
        $avg = ($r.Probes | Measure-Object -Average).Average
    }
    Emit ("FGCC proberun run={0} prompt={1} completion={2} prefill_tps={3:F2} decode_tps={4:F2} wall_s={5:F1} probes={6} probe_max_ms={7:F1} probe_avg_ms={8:F1}" -f
        $i, $r.Prompt, $r.Completion, $r.PrefillTps, $r.DecodeTps, $r.Wall,
        $r.Probes.Count, $max, $avg)
}
$bestControlDecode = ($controls | Measure-Object -Property DecodeTps -Maximum).Maximum
$bestProbeDecode = ($probes | Measure-Object -Property DecodeTps -Maximum).Maximum
$bestControlPrefill = ($controls | Measure-Object -Property PrefillTps -Maximum).Maximum
$bestProbePrefill = ($probes | Measure-Object -Property PrefillTps -Maximum).Maximum
$decodeDelta = if ($bestControlDecode -gt 0) {
    100.0 * ($bestProbeDecode - $bestControlDecode) / $bestControlDecode } else { 0.0 }
$prefillDelta = if ($bestControlPrefill -gt 0) {
    100.0 * ($bestProbePrefill - $bestControlPrefill) / $bestControlPrefill } else { 0.0 }
Emit ("FGCC best control_decode={0:F2} probe_decode={1:F2} delta_pct={2:F2}" -f
    $bestControlDecode, $bestProbeDecode, $decodeDelta)
Emit ("FGCC best control_prefill={0:F2} probe_prefill={1:F2} delta_pct={2:F2}" -f
    $bestControlPrefill, $bestProbePrefill, $prefillDelta)

if ($OutFile) { $log | Set-Content -Path $OutFile -Encoding ascii }
