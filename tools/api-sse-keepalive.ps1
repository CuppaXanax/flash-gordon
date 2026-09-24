#requires -Version 7.0
<#
SSE keep-alive evidence: start a streaming completion with a long prefill
(~26K tokens) and capture raw SSE bytes with arrival timestamps.  Counts the
`: keep-alive` comments and their spacing while the engine is prefilling.

FGSSE lines are the citable output.
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:FG_BASE_URL) { $env:FG_BASE_URL } else { "http://192.168.42.42:8080" }),
    [int]$PromptTokens = 26000,
    [int]$MaxTokens = 8,
    [int]$TimeoutSec = 240,
    [string]$RawOut = "",
    [string]$OutFile = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$log = [Collections.Generic.List[string]]::new()
function Emit([string]$Line) { $log.Add($Line); Write-Host $Line }

$uri = [Uri]$BaseUrl
$builder = [Text.StringBuilder]::new()
$sentence = "The architecture of the system is described in this section. "
while (($builder.Length / 4) -lt $PromptTokens) { [void]$builder.Append($sentence) }
$body = @{ model = "Qwen3.8-Flash-Next"
    messages = @(@{ role = "user"; content = $builder.ToString() })
    temperature = 0.0; top_p = 1.0; top_k = 1
    max_tokens = $MaxTokens; stream = $true } | ConvertTo-Json -Depth 10 -Compress
$bodyBytes = [Text.Encoding]::UTF8.GetBytes($body)
$head = "POST /v1/chat/completions HTTP/1.1`r`nHost: $($uri.Host)`r`n" +
        "Content-Type: application/json`r`nContent-Length: $($bodyBytes.Length)`r`n`r`n"

$client = [System.Net.Sockets.TcpClient]::new()
$client.Connect($uri.Host, $uri.Port)
$client.ReceiveTimeout = 2000
$stream = $client.GetStream()
$sw = [Diagnostics.Stopwatch]::StartNew()
$stream.Write([Text.Encoding]::ASCII.GetBytes($head), 0, $head.Length)
$stream.Write($bodyBytes, 0, $bodyBytes.Length)
$stream.Flush()

$all = [Text.StringBuilder]::new()
$comments = [Collections.Generic.List[double]]::new()
$firstDataMs = -1.0
$lastComment = 0.0
$buffer = New-Object byte[] 16384
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    $got = $false
    try {
        while ($stream.DataAvailable) {
            $read = $stream.Read($buffer, 0, $buffer.Length)
            if ($read -le 0) { break }
            $got = $true
            $text = [Text.Encoding]::UTF8.GetString($buffer, 0, $read)
            [void]$all.Append($text)
            if (($text -match "data: ") -and $firstDataMs -lt 0) {
                $firstDataMs = $sw.Elapsed.TotalMilliseconds
            }
            # Count only SSE comment lines (`\n: keep-alive`), not the
            # `Connection: keep-alive` response header.
            $scan = $all.ToString()
            $found = 0
            $idx = $scan.IndexOf("`n: keep-alive")
            while ($idx -ge 0) { $found++; $idx = $scan.IndexOf("`n: keep-alive", $idx + 1) }
            while ($comments.Count -lt $found) { $comments.Add($sw.Elapsed.TotalMilliseconds) }
        }
    } catch { }
    if ($all.ToString().Contains("data: [DONE]")) { break }
    if (-not $got) { Start-Sleep -Milliseconds 20 }
}
$sw.Stop()
$raw = $all.ToString()
if ($RawOut) { $raw | Set-Content -Path $RawOut -Encoding ascii }

Emit ("FGSSE wall_s={0:F1} bytes={1} first_data_ms={2:F0} comment_count={3}" -f
    $sw.Elapsed.TotalSeconds, $raw.Length, $firstDataMs, $comments.Count)
for ($i = 0; $i -lt $comments.Count; $i++) {
    $gap = if ($i -eq 0) { $comments[$i] } else { $comments[$i] - $comments[$i - 1] }
    Emit ("FGSSE comment index={0} at_ms={1:F0} since_previous_ms={2:F0}" -f
        $i, $comments[$i], $gap)
}
Emit ("FGSSE done={0}" -f $raw.Contains("data: [DONE]"))

if ($OutFile) { $log | Set-Content -Path $OutFile -Encoding ascii }
