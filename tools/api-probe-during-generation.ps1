#requires -Version 7.0
<#
Probe-latency-during-generation evidence for the API front-end split.

Starts one long chat completion (text prefill+decode or a 512px vision turn),
then hammers /health and /v1/models from separate connections while it runs,
plus one second chat request while the engine is busy.  Records per-probe
latencies, the busy-chat status/Retry-After and the generation wall time.

M1 refused the second chat with 503; M2 admits it to the bounded queue, so the
10 s busy-chat probe client times out with `queued=1` (the request is waiting
for its turn, served after the generation; see tools/api-queue.ps1 for the
dedicated admission evidence).

FGPROBE lines are the citable output.
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:FG_BASE_URL) { $env:FG_BASE_URL } else { "http://192.168.42.42:8080" }),
    [ValidateSet("text", "vision")]
    [string]$Mode = "text",
    [int]$PromptTokens = 12000,
    [int]$MaxTokens = 192,
    [int]$ProbeIntervalMs = 250,
    [string]$VisionBody = "$env:TEMP\opencode\vision512-body.json",
    [string]$OutFile = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$log = [Collections.Generic.List[string]]::new()
function Emit([string]$Line) { $log.Add($Line); Write-Host $Line }

function New-Client([int]$TimeoutSec) {
    $handler = [System.Net.Http.SocketsHttpHandler]::new()
    $handler.PooledConnectionIdleTimeout = [TimeSpan]::FromSeconds(5)
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    return $client
}

$longClient = New-Client 3600
$probeClient = New-Client 10

function Invoke-Probe([string]$Path) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = $probeClient.GetAsync("$BaseUrl$Path").GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $sw.Stop()
        return [pscustomobject]@{ Ms = $sw.Elapsed.TotalMilliseconds; Status = [int]$resp.StatusCode
            Body = $text; Error = $null }
    } catch {
        $sw.Stop()
        return [pscustomobject]@{ Ms = $sw.Elapsed.TotalMilliseconds; Status = 0; Body = ""
            Error = $_.Exception.GetBaseException().Message }
    }
}

function Start-BusyChatProbe {
    $small = @{ model = "Qwen3.8-Flash-Next"
        messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1; stream = $false } |
        ConvertTo-Json -Depth 10 -Compress
    $content = [System.Net.Http.StringContent]::new($small, [Text.Encoding]::UTF8,
        "application/json")
    $client = New-Client 10
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # Fire-and-record: the sampler must not block on it (a queued request can
    # wait out a whole generation; in M1 the same probe returned 503 at once).
    $task = $client.PostAsync("$BaseUrl/v1/chat/completions", $content)
    return [pscustomobject]@{ Task = $task; Sw = $sw; Client = $client }
}

function Finish-BusyChatProbe($job) {
    try {
        $resp = $job.Task.GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $job.Sw.Stop()
        $retry = if ($resp.Headers.Contains("Retry-After")) {
            ($resp.Headers.GetValues("Retry-After") -join ",")
        } else { "" }
        return [pscustomobject]@{ Ms = $job.Sw.Elapsed.TotalMilliseconds
            Status = [int]$resp.StatusCode; Body = $text; RetryAfter = $retry
            Queued = $false }
    } catch {
        $job.Sw.Stop()
        $base = $_.Exception.GetBaseException()
        # M2: a second chat while busy is queued, not refused, so the 10 s
        # probe client times out waiting for its turn - that is admission.
        $queued = ($base -is [System.Threading.Tasks.TaskCanceledException]) -or
                  ($base.Message -match 'aborted|timed out')
        return [pscustomobject]@{ Ms = $job.Sw.Elapsed.TotalMilliseconds; Status = 0
            Body = $base.Message; RetryAfter = ""; Queued = $queued }
    } finally { $job.Client.Dispose() }
}

# --- build the long request -------------------------------------------------
if ($Mode -eq "vision") {
    if (-not (Test-Path -LiteralPath $VisionBody)) { throw "missing $VisionBody" }
    $body = [IO.File]::ReadAllText($VisionBody)
} else {
    $builder = [Text.StringBuilder]::new()
    $sentence = "The architecture of the system is described in this section. "
    while (($builder.Length / 4) -lt $PromptTokens) { [void]$builder.Append($sentence) }
    $body = @{ model = "Qwen3.8-Flash-Next"
        messages = @(@{ role = "user"; content = $builder.ToString() })
        temperature = 0.0; top_p = 1.0; top_k = 1
        max_tokens = $MaxTokens; stream = $false } | ConvertTo-Json -Depth 10 -Compress
}

# --- idle baseline ----------------------------------------------------------
$idle = @()
for ($i = 0; $i -lt 10; $i++) {
    $p = Invoke-Probe "/health"
    $idle += $p.Ms
}
Emit ("FGPROBE idle_health n={0} max_ms={1:F1} avg_ms={2:F1}" -f $idle.Count,
    ($idle | Measure-Object -Maximum).Maximum, ($idle | Measure-Object -Average).Average)

# --- start the long generation ---------------------------------------------
$content = [System.Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, "application/json")
$wall = [Diagnostics.Stopwatch]::StartNew()
$task = $longClient.PostAsync("$BaseUrl/v1/chat/completions", $content)

# Wait for the engine to go busy (or the request to finish early).
$busySeen = $false
for ($i = 0; $i -lt 200; $i++) {
    if ($task.IsCompleted) { break }
    Start-Sleep -Milliseconds 100
    $p = Invoke-Probe "/health"
    if ($p.Body -match '"busy":true') { $busySeen = $true; break }
}
Emit ("FGPROBE busy_seen={0}" -f $busySeen)

# --- probe while generating -------------------------------------------------
$health = @()
$models = @()
$errors = 0
$busyChatJob = $null
$probeIndex = 0
while (-not $task.IsCompleted) {
    $p = Invoke-Probe "/health"
    $health += $p.Ms
    if ($p.Error) { $errors++ }
    if (($probeIndex % 8) -eq 0) {
        $m = Invoke-Probe "/v1/models"
        $models += $m.Ms
        if ($m.Error) { $errors++ }
    }
    if ($null -eq $busyChatJob -and $busySeen) { $busyChatJob = Start-BusyChatProbe }
    $probeIndex++
    Start-Sleep -Milliseconds $ProbeIntervalMs
}
$wall.Stop()
$busyChat = if ($busyChatJob) { Finish-BusyChatProbe $busyChatJob } else { $null }

function Get-Stats([double[]]$Values) {
    if ($null -eq $Values -or $Values.Count -eq 0) { return @(0.0, 0.0, 0.0) }
    $sorted = @($Values | Sort-Object)
    $p95 = $sorted[[Math]::Min($sorted.Count - 1, [Math]::Floor($sorted.Count * 0.95))]
    return @(($sorted | Measure-Object -Maximum).Maximum,
              ($Values | Measure-Object -Average).Average, $p95)
}

$hs = Get-Stats $health
$ms = Get-Stats $models
Emit ("FGPROBE probe health n={0} max_ms={1:F1} avg_ms={2:F1} p95_ms={3:F1} errors={4}" -f
    $health.Count, $hs[0], $hs[1], $hs[2], $errors)
Emit ("FGPROBE probe models n={0} max_ms={1:F1} avg_ms={2:F1} p95_ms={3:F1}" -f
    $models.Count, $ms[0], $ms[1], $ms[2])
if ($busyChat) {
    Emit ("FGPROBE busy_chat status={0} latency_ms={1:F1} retry_after='{2}' queued={3} note='{4}'" -f
        $busyChat.Status, $busyChat.Ms, $busyChat.RetryAfter, [int]$busyChat.Queued,
        $busyChat.Body.Substring(0, [Math]::Min(60, $busyChat.Body.Length)))
}

# --- generation result ------------------------------------------------------
$status = 0; $promptTokens = 0; $completionTokens = 0; $note = ""
try {
    $resp = $task.GetAwaiter().GetResult()
    $status = [int]$resp.StatusCode
    $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if ($status -eq 200) {
        $json = $text | ConvertFrom-Json
        $promptTokens = [int]$json.usage.prompt_tokens
        $completionTokens = [int]$json.usage.completion_tokens
    } else {
        $note = $text.Substring(0, [Math]::Min(160, $text.Length))
    }
} catch { $status = 0; $note = $_.Exception.GetBaseException().Message }
Emit ("FGPROBE generation mode={0} http={1} wall_s={2:F1} prompt_tokens={3} completion_tokens={4} note='{5}'" -f
    $Mode, $status, $wall.Elapsed.TotalSeconds, $promptTokens, $completionTokens, $note)

if ($OutFile) { $log | Set-Content -Path $OutFile -Encoding ascii }
