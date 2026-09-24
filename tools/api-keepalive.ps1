#requires -Version 7.0
<#
Raw-TCP keep-alive evidence: sequential requests on one connection.

1. GET /health            -> expect 200, Connection: keep-alive, content-length body
2. GET /v1/models         -> same connection, second response (persistent)
3. GET /health Connection: close -> expect 200, Connection: close, server closes
4. malformed request      -> expect 400 and close
FGKA lines are the citable output.
#>
[CmdletBinding()]
param(
    [string]$HostName = "192.168.42.42",
    [int]$Port = 8080,
    [string]$OutFile = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$log = [Collections.Generic.List[string]]::new()
function Emit([string]$Line) { $log.Add($Line); Write-Host $Line }

function New-Connection {
    $client = [System.Net.Sockets.TcpClient]::new()
    $client.Connect($HostName, $Port)
    $client.ReceiveTimeout = 5000
    $client.SendTimeout = 5000
    return $client
}

function Send-Line($client, [string]$Text) {
    $bytes = [Text.Encoding]::ASCII.GetBytes($Text)
    $stream = $client.GetStream()
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Flush()
}

function Read-Response($client) {
    # Read one HTTP/1.1 response.  Handles Content-Length and chunked bodies;
    # returns the raw text plus a parsed status and connection header.
    $stream = $client.GetStream()
    $buffer = [Collections.Generic.List[byte]]::new()
    $one = New-Object byte[] 4096
    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not $stream.DataAvailable) {
            if ($client.Available -eq 0) {
                if ($buffer.Count -gt 0 -and $client.Client.Poll(10000, [Net.Sockets.SelectMode]::SelectRead) -eq $false) {
                    break
                }
                Start-Sleep -Milliseconds 20
                continue
            }
        }
        $read = $stream.Read($one, 0, $one.Length)
        if ($read -le 0) { break }
        for ($i = 0; $i -lt $read; $i++) { $buffer.Add($one[$i]) }
        $text = [Text.Encoding]::UTF8.GetString($buffer.ToArray())
        if ($text -match "\r\n0\r\n\r\n$") { break }
        if ($text -match "\r\n\r\n") {
            $headerEnd = $text.IndexOf("`r`n`r`n")
            $head = $text.Substring(0, $headerEnd)
            $m = [regex]::Match($head, "Content-Length:\s*(\d+)", "IgnoreCase")
            if ($m.Success) {
                $need = $headerEnd + 4 + [int]$m.Groups[1].Value
                if ($buffer.Count -ge $need) { break }
            }
        }
    }
    $raw = [Text.Encoding]::UTF8.GetString($buffer.ToArray())
    $status = 0
    $sm = [regex]::Match($raw, "^HTTP/1\.1 (\d{3})")
    if ($sm.Success) { $status = [int]$sm.Groups[1].Value }
    $connection = ""
    $cm = [regex]::Match($raw, "Connection:\s*(\S+)", "IgnoreCase")
    if ($cm.Success) { $connection = $cm.Groups[1].Value.Trim() }
    return [pscustomobject]@{ Raw = $raw; Status = $status; Connection = $connection }
}

$client = New-Connection
$sw = [Diagnostics.Stopwatch]::StartNew()
Send-Line $client "GET /health HTTP/1.1`r`nHost: fg`r`n`r`n"
$one = Read-Response $client
$sw.Stop()
Emit ("FGKA request=1 path=/health status={0} connection='{1}' body_has_status={2} ms={3:F1}" -f
    $one.Status, $one.Connection, ($one.Raw -match '"status":"ok"'), $sw.Elapsed.TotalMilliseconds)

$sw.Restart()
Send-Line $client "GET /v1/models HTTP/1.1`r`nHost: fg`r`n`r`n"
$two = Read-Response $client
$sw.Stop()
Emit ("FGKA request=2 path=/v1/models same_connection=true status={0} connection='{1}' body_has_model={2} ms={3:F1}" -f
    $two.Status, $two.Connection, ($two.Raw -match 'Qwen3.8-Flash-Next'), $sw.Elapsed.TotalMilliseconds)

$sw.Restart()
Send-Line $client "GET /health HTTP/1.1`r`nHost: fg`r`nConnection: close`r`n`r`n"
$three = Read-Response $client
$sw.Stop()
$closed = $false
try {
    $client.Client.Poll(3000000, [Net.Sockets.SelectMode]::SelectRead) | Out-Null
    $closed = $client.Client.Poll(0, [Net.Sockets.SelectMode]::SelectRead) -and
              ($client.Client.Available -eq 0)
} catch { $closed = $true }
Emit ("FGKA request=3 path=/health close_requested=true status={0} connection='{1}' server_closed={2} ms={3:F1}" -f
    $three.Status, $three.Connection, $closed, $sw.Elapsed.TotalMilliseconds)
$client.Close()

$bad = New-Connection
Send-Line $bad "GARBAGE`r`n`r`n"
$badResponse = Read-Response $bad
Emit ("FGKA malformed status={0} close_after={1}" -f $badResponse.Status,
    ($badResponse.Connection -eq "close"))
$bad.Close()

if ($OutFile) { $log | Set-Content -Path $OutFile -Encoding ascii }
