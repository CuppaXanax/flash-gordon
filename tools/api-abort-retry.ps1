#requires -Version 7.0
<#
Abort/retry evidence: send a long streaming prefill over a raw TCP connection,
read SSE for a few seconds, then close the socket (hard client disconnect).
Measure how long the engine takes to free up (busy:false), then retry the same
prompt and report the prefix-cache headers (frontier preserved).

FGABORT lines are the citable output.
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:FG_BASE_URL) { $env:FG_BASE_URL } else { "http://192.168.42.42:8080" }),
    [int]$PromptTokens = 12000,
    [int]$StreamSeconds = 4,
    [string]$OutFile = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$log = [Collections.Generic.List[string]]::new()
function Emit([string]$Line) { $log.Add($Line); Write-Host $Line }

$uri = [Uri]$BaseUrl
function New-Client([int]$TimeoutSec) {
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    return $client
}
$probe = New-Client 10

function Get-Busy {
    try {
        $resp = $probe.GetAsync("$BaseUrl/health").GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return $text -match '"busy":true'
    } catch { return $false }
}

$builder = [Text.StringBuilder]::new()
$sentence = "The architecture of the system is described in this section. "
while (($builder.Length / 4) -lt $PromptTokens) { [void]$builder.Append($sentence) }
$prompt = $builder.ToString()
$body = @{ model = "Qwen3.8-Flash-Next"
    messages = @(@{ role = "user"; content = $prompt })
    temperature = 0.0; top_p = 1.0; top_k = 1
    max_tokens = 64; stream = $true } | ConvertTo-Json -Depth 10 -Compress
$bodyBytes = [Text.Encoding]::UTF8.GetBytes($body)
$head = "POST /v1/chat/completions HTTP/1.1`r`nHost: $($uri.Host)`r`n" +
        "Content-Type: application/json`r`nContent-Length: $($bodyBytes.Length)`r`n`r`n"

$client = [System.Net.Sockets.TcpClient]::new()
$client.Connect($uri.Host, $uri.Port)
$client.ReceiveTimeout = 2000
$stream = $client.GetStream()
$stream.Write([Text.Encoding]::ASCII.GetBytes($head), 0, $head.Length)
$stream.Write($bodyBytes, 0, $bodyBytes.Length)
$stream.Flush()

$sw = [Diagnostics.Stopwatch]::StartNew()
$bytesRead = 0
$buffer = New-Object byte[] 4096
while ($sw.Elapsed.TotalSeconds -lt $StreamSeconds) {
    try {
        if ($stream.DataAvailable) {
            $read = $stream.Read($buffer, 0, $buffer.Length)
            if ($read -le 0) { break }
            $bytesRead += $read
        } else {
            Start-Sleep -Milliseconds 25
        }
    } catch { break }
}
Emit ("FGABORT stream_s={0:F1} bytes_read={1} closing=1" -f $sw.Elapsed.TotalSeconds, $bytesRead)

$cancelAt = [Diagnostics.Stopwatch]::StartNew()
$client.Close()   # FIN/RST: the server must see client_gone

$freed = -1.0
for ($i = 0; $i -lt 200; $i++) {
    Start-Sleep -Milliseconds 100
    if (-not (Get-Busy)) { $freed = $cancelAt.Elapsed.TotalSeconds; break }
}
Emit ("FGABORT engine_free_s={0:F1}" -f $freed)

# Retry the same prompt: the interrupted prefill frontier must still be there.
$retryClient = New-Client 1800
$retryBody = @{ model = "Qwen3.8-Flash-Next"
    messages = @(@{ role = "user"; content = $prompt })
    temperature = 0.0; top_p = 1.0; top_k = 1
    max_tokens = 4; stream = $false } | ConvertTo-Json -Depth 10 -Compress
$retryContent = [System.Net.Http.StringContent]::new($retryBody, [Text.Encoding]::UTF8,
    "application/json")
$retrySw = [Diagnostics.Stopwatch]::StartNew()
$retry = $retryClient.PostAsync("$BaseUrl/v1/chat/completions", $retryContent).
    GetAwaiter().GetResult()
$retryText = $retry.Content.ReadAsStringAsync().GetAwaiter().GetResult()
$retrySw.Stop()
$cache = if ($retry.Headers.Contains("X-Flash-Gordon-Prefix-Cache")) {
    ($retry.Headers.GetValues("X-Flash-Gordon-Prefix-Cache") -join ",") } else { "" }
$reused = if ($retry.Headers.Contains("X-Flash-Gordon-Reused-Tokens")) {
    ($retry.Headers.GetValues("X-Flash-Gordon-Reused-Tokens") -join ",") } else { "" }
$prefilled = if ($retry.Headers.Contains("X-Flash-Gordon-Prefilled-Tokens")) {
    ($retry.Headers.GetValues("X-Flash-Gordon-Prefilled-Tokens") -join ",") } else { "" }
Emit ("FGABORT retry http={0} prefix_cache={1} reused_tokens={2} prefilled_tokens={3} wall_s={4:F1}" -f
    [int]$retry.StatusCode, $cache, $reused, $prefilled, $retrySw.Elapsed.TotalSeconds)
Emit ("FGABORT retry_body_ok={0}" -f ($retry.StatusCode -eq 200 -and $retryText.Length -gt 0))

if ($OutFile) { $log | Set-Content -Path $OutFile -Encoding ascii }
