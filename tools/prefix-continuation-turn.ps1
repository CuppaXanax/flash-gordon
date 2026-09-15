#requires -Version 7.0
# Growing-conversation prefix-continuation measurement (non-streaming; the
# X-Flash-Gordon-* metrics headers are only emitted on the non-streaming path).
[CmdletBinding()]
param(
    [string]$ApiUrl = $(if ($env:FG_API_URL) { $env:FG_API_URL } else { "http://192.0.2.42:8080/v1/chat/completions" }),
    [string]$Model = "Qwen3.8-Flash-Next",
    [int]$Turns = 4,
    [int]$FillerRepeats = 0,
    [int]$MaxTokens = 24,
    [switch]$ForceColdBeforeFinal,
    [switch]$NoFiller,
    [string]$OutPath = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-Filler([int]$repeats) {
    if ($repeats -le 0) { return "" }
    $line = "The quick brown fox jumps over the lazy dog. "
    return ($line * $repeats)
}

function Get-HeaderValue($headers, [string]$name) {
    if ($headers.ContainsKey($name)) { $v = $headers[$name]; if ($v -is [array]) { return $v[0] } return $v }
    return ""
}

$messages = [System.Collections.Generic.List[object]]::new()
$messages.Add(@{ role = "system"; content = "You are a terse assistant. Answer every question with one short sentence." })

$results = [System.Collections.Generic.List[object]]::new()
$fillerPerTurn = if ($NoFiller) { 0 } else { $FillerRepeats }

for ($turn = 1; $turn -le $Turns; $turn++) {
    $filler = New-Filler $fillerPerTurn
    $question = "Question $turn`: in one short sentence, what is the number $($turn * 7) plus 3?"
    $messages.Add(@{ role = "user"; content = "/no_think $filler$question" })

    if ($ForceColdBeforeFinal -and $turn -eq $Turns) {
        $tiny = @{ model = $Model; messages = @(@{ role = "user"; content = "hi" }); temperature = 0.0; max_tokens = 1; stream = $false } | ConvertTo-Json -Depth 8 -Compress
        Invoke-WebRequest -Uri $ApiUrl -Method Post -ContentType "application/json" -Body $tiny -TimeoutSec 1800 | Out-Null
        Start-Sleep -Milliseconds 200
    }

    $body = @{
        model = $Model
        messages = $messages.ToArray()
        temperature = 0.0
        top_p = 1.0
        top_k = 1
        max_tokens = $MaxTokens
        stream = $false
    } | ConvertTo-Json -Depth 8 -Compress

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-WebRequest -Uri $ApiUrl -Method Post -ContentType "application/json" -Body $body -TimeoutSec 3600
    $sw.Stop()
    $p = $r.Content | ConvertFrom-Json
    $content = $p.choices[0].message.content
    if ($null -eq $content) { $content = "" }

    $promptTokens = [int](Get-HeaderValue $r.Headers "X-Flash-Gordon-Prompt-Tokens")
    $prefilledTokens = [int](Get-HeaderValue $r.Headers "X-Flash-Gordon-Prefilled-Tokens")
    $reusedTokens = [int](Get-HeaderValue $r.Headers "X-Flash-Gordon-Reused-Tokens")
    $prefixCache = Get-HeaderValue $r.Headers "X-Flash-Gordon-Prefix-Cache"
    $resetReason = Get-HeaderValue $r.Headers "X-Flash-Gordon-Reset-Reason"
    $contextTokens = [int](Get-HeaderValue $r.Headers "X-Flash-Gordon-Context-Tokens")
    $prefillSeconds = [double](Get-HeaderValue $r.Headers "X-Flash-Gordon-Prefill-Seconds")
    $decodeSeconds = [double](Get-HeaderValue $r.Headers "X-Flash-Gordon-Decode-Seconds")
    $completionTokens = [int](Get-HeaderValue $r.Headers "X-Flash-Gordon-Completion-Tokens")
    $ttft = $prefillSeconds
    if ($completionTokens -gt 0 -and $decodeSeconds -gt 0) { $ttft += $decodeSeconds / $completionTokens }

    $results.Add([pscustomobject]@{
        turn      = $turn
        prompt    = $promptTokens
        prefilled = $prefilledTokens
        reused    = $reusedTokens
        cache     = $prefixCache
        reset     = $resetReason
        prefill_s = [math]::Round($prefillSeconds, 3)
        ttft_s    = [math]::Round($ttft, 3)
        wall_s    = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        completion = $completionTokens
        content   = $content.Trim()
    })
    $messages.Add(@{ role = "assistant"; content = $content })
    Write-Host ("turn={0} prompt={1} prefilled={2} reused={3} cache={4} reset={5} prefill_s={6} ttft_s={7} wall_s={8} completion={9} content=[{10}]" -f `
        $turn, $promptTokens, $prefilledTokens, $reusedTokens, $prefixCache, $resetReason, `
        ([math]::Round($prefillSeconds,3)), ([math]::Round($ttft,3)), ([math]::Round($sw.Elapsed.TotalSeconds,3)), $completionTokens, $content.Trim())
}

if ($OutPath) {
    $results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutPath
}
