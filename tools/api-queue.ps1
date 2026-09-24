#requires -Version 7.0
<#
API M2 admission-queue evidence.

Starts one long generation to keep the engine busy, then exercises the
bounded FIFO admission policy:
  - a queued chat request is admitted (no immediate 503) and served FIFO;
  - a burst over the bound gets 429 + Retry-After: 1 (type queue_full);
  - a queued client that disconnects is canceled and its slot is released.

Slots: bound 4 = long generation + queued request + raw cancel candidate +
one admitted burst request; the other burst requests are 429.  After the raw
client drops, one more request must be admitted, proving the cancel sweep
released the slot.

FGQUEUE lines are the citable output.
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:FG_BASE_URL) { $env:FG_BASE_URL } else { "http://192.168.42.42:8080" }),
    [int]$PromptTokens = 8000,
    [int]$MaxTokens = 64,
    [int]$Burst = 4,
    [string]$OutFile = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$log = [Collections.Generic.List[string]]::new()
function Emit([string]$Line) { $log.Add($Line); Write-Host $Line }

$smallBody = @{ model = "Qwen3.8-Flash-Next"
    messages = @(@{ role = "user"; content = "hi" })
    temperature = 0.0; top_p = 1.0; top_k = 1
    max_tokens = 1; stream = $false } | ConvertTo-Json -Depth 10 -Compress

function New-Client([int]$TimeoutSec) {
    $handler = [System.Net.Http.SocketsHttpHandler]::new()
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    return $client
}

function Start-Small($client) {
    $content = [System.Net.Http.StringContent]::new($smallBody, [Text.Encoding]::UTF8,
        "application/json")
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $task = $client.PostAsync("$BaseUrl/v1/chat/completions", $content)
    return [pscustomobject]@{ Task = $task; Sw = $sw }
}

function Finish-Small($job) {
    try {
        $resp = $job.Task.GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $job.Sw.Stop()
        $retry = if ($resp.Headers.Contains("Retry-After")) {
            ($resp.Headers.GetValues("Retry-After") -join ",") } else { "" }
        return [pscustomobject]@{ Ms = $job.Sw.Elapsed.TotalMilliseconds
            Status = [int]$resp.StatusCode; RetryAfter = $retry; Body = $text }
    } catch {
        $job.Sw.Stop()
        return [pscustomobject]@{ Ms = $job.Sw.Elapsed.TotalMilliseconds; Status = 0
            RetryAfter = ""; Body = $_.Exception.GetBaseException().Message }
    }
}

function Invoke-Probe([string]$Path) {
    $c = New-Client 10
    try {
        $resp = $c.GetAsync("$BaseUrl$Path").GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return [pscustomobject]@{ Status = [int]$resp.StatusCode; Body = $text }
    } catch {
        return [pscustomobject]@{ Status = 0; Body = $_.Exception.GetBaseException().Message }
    } finally { $c.Dispose() }
}

# --- long generation ---------------------------------------------------------
$builder = [Text.StringBuilder]::new()
$sentence = "The architecture of the system is described in this section. "
while (($builder.Length / 4) -lt $PromptTokens) { [void]$builder.Append($sentence) }
$longBody = @{ model = "Qwen3.8-Flash-Next"
    messages = @(@{ role = "user"; content = $builder.ToString() })
    temperature = 0.0; top_p = 1.0; top_k = 1
    max_tokens = $MaxTokens; stream = $false } | ConvertTo-Json -Depth 10 -Compress

$longClient = New-Client 3600
$content = [System.Net.Http.StringContent]::new($longBody, [Text.Encoding]::UTF8,
    "application/json")
$wall = [Diagnostics.Stopwatch]::StartNew()
$longTask = $longClient.PostAsync("$BaseUrl/v1/chat/completions", $content)

$busySeen = $false
for ($i = 0; $i -lt 300; $i++) {
    if ($longTask.IsCompleted) { break }
    Start-Sleep -Milliseconds 100
    if ((Invoke-Probe "/health").Body -match '"busy":true') { $busySeen = $true; break }
}
Emit ("FGQUEUE busy_seen={0}" -f $busySeen)
if (-not $busySeen) {
    Emit "FGQUEUE ABORT no busy generation"
    if ($OutFile) { $log | Set-Content $OutFile -Encoding ascii }
    exit 2
}

# --- slot 2: a queued request is admitted and served FIFO --------------------
$queuedClient = New-Client 3600
$queuedJob = Start-Small $queuedClient

# --- slot 3: raw cancel candidate --------------------------------------------
$uri = [Uri]$BaseUrl
$tcp = [Net.Sockets.TcpClient]::new()
$tcp.Connect($uri.Host, $uri.Port)
$bodyBytes = [Text.Encoding]::UTF8.GetBytes($smallBody)
$head = "POST /v1/chat/completions HTTP/1.1`r`nHost: fg`r`n" +
        "Content-Type: application/json`r`nContent-Length: $($bodyBytes.Length)`r`n`r`n"
$stream = $tcp.GetStream()
$stream.Write([Text.Encoding]::ASCII.GetBytes($head))
$stream.Write($bodyBytes)
$stream.Flush()
Start-Sleep -Milliseconds 250

# --- slot 4: one admitted burst request; the rest are over the bound ---------
$burstJobs = @()
for ($i = 0; $i -lt $Burst; $i++) {
    $c = New-Client 3600
    $burstJobs += Start-Small $c
}
Start-Sleep -Milliseconds 900
$admitted = 0; $rejected = 0; $typeFull = 0; $retrySeen = 0
$pending = @()
foreach ($job in $burstJobs) {
    if ($job.Task.IsCompleted) {
        $r = Finish-Small $job
        if ($r.Status -eq 429) {
            $rejected++
            if ($r.RetryAfter -eq "1") { $retrySeen++ }
            if ($r.Body -match '"type":"queue_full"') { $typeFull++ }
        } elseif ($r.Status -eq 200) { $admitted++ }
    } else { $pending += $job; $admitted++ }
}
Emit ("FGQUEUE burst n={0} admitted={1} rejected_429={2} retry_after_1={3} queue_full={4}" -f
    $Burst, $admitted, $rejected, $retrySeen, $typeFull)

# --- the raw client drops while queued: its slot must be released ------------
$tcp.Close()
$canaryJob = $null
$slotReleased = $false
$canaryClient = New-Client 3600
for ($i = 0; $i -lt 12 -and -not $slotReleased; $i++) {
    Start-Sleep -Milliseconds 300
    $job = Start-Small $canaryClient
    Start-Sleep -Milliseconds 1200
    if (-not $job.Task.IsCompleted) {
        $slotReleased = $true
        $canaryJob = $job
        break
    }
    $r = Finish-Small $job
    if ($r.Status -eq 200) { $slotReleased = $true; $canaryJob = $job; break }
}
Emit ("FGQUEUE cancel slot_released={0}" -f ([int]$slotReleased))

# --- drain: wait for the long generation, then collect the queued results ----
$idle = $false
for ($i = 0; $i -lt 900; $i++) {
    Start-Sleep -Milliseconds 500
    if ($longTask.IsCompleted) { break }
    if ((Invoke-Probe "/health").Body -match '"busy":false') { $idle = $true; break }
}
$wall.Stop()
$status = 0; $completion = 0; $note = ""
try {
    $resp = $longTask.GetAwaiter().GetResult()
    $status = [int]$resp.StatusCode
    $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if ($status -eq 200) { $completion = [int](($text | ConvertFrom-Json).usage.completion_tokens) }
    else { $note = $text.Substring(0, [Math]::Min(160, $text.Length)) }
} catch { $status = 0; $note = $_.Exception.GetBaseException().Message }
Emit ("FGQUEUE generation http={0} wall_s={1:F1} completion_tokens={2} note='{3}'" -f
    $status, $wall.Elapsed.TotalSeconds, $completion, $note)

$queuedResult = Finish-Small $queuedJob
Emit ("FGQUEUE queued_request status={0} wall_ms={1:F0}" -f $queuedResult.Status,
    $queuedResult.Ms)
foreach ($job in $pending) {
    $r = Finish-Small $job
    Emit ("FGQUEUE burst_admitted status={0} wall_ms={1:F0}" -f $r.Status, $r.Ms)
}
if ($canaryJob) {
    $canaryResult = Finish-Small $canaryJob
    Emit ("FGQUEUE cancel canary_status={0} wall_ms={1:F0}" -f $canaryResult.Status,
        $canaryResult.Ms)
}
$idleAgain = (Invoke-Probe "/health").Body -match '"busy":false'
Emit ("FGQUEUE idle_again={0}" -f ([int]($idle -or $idleAgain)))

if ($OutFile) { $log | Set-Content -Path $OutFile -Encoding ascii }
