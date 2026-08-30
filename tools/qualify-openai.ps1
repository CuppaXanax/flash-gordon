param(
    [string]$BaseUrl = "http://192.168.42.42:8080/v1",
    [string]$Model = "Qwen3.8-Flash-Next",
    [int]$WarmupRuns = 1,
    [int]$MeasuredRuns = 5
)

$ErrorActionPreference = "Stop"
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$BaselinePath = Join-Path (Split-Path $PSScriptRoot -Parent) "qualification-baseline.json"
$baseline = Get-Content -Raw -LiteralPath $BaselinePath | ConvertFrom-Json
$MinimumDecodeTps = [double]$baseline.decode.minimum_median_tps
if ($Model -ne [string]$baseline.model) {
    throw "Requested model $Model does not match frozen baseline model $($baseline.model)."
}

if ($WarmupRuns -lt 0 -or $MeasuredRuns -lt 3 -or ($MeasuredRuns % 2) -eq 0) {
    throw "WarmupRuns must be non-negative and MeasuredRuns must be an odd value of at least 3."
}

function Invoke-JsonRequest {
    param([hashtable]$Body)

    Invoke-WebRequest -Uri "$BaseUrl/chat/completions" -Method Post `
        -ContentType "application/json" `
        -Body ($Body | ConvertTo-Json -Depth 20 -Compress) `
        -TimeoutSec 900
}

function Get-HeaderDouble {
    param($Response, [string]$Name)

    $value = [string]$Response.Headers[$Name]
    if (-not $value) {
        throw "Response did not include required qualification header $Name."
    }
    [double]::Parse($value, $InvariantCulture)
}

function Get-Median {
    param([double[]]$Values)

    $ordered = @($Values | Sort-Object)
    $ordered[[int][Math]::Floor($ordered.Count / 2)]
}

$models = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 30
if (-not $models.data -or $models.data[0].id -ne $Model) {
    throw "Expected model $Model from /v1/models."
}

$hello = @{
    model = $Model
    messages = @(
        @{ role = "system"; content = "Answer directly and briefly." },
        @{ role = "user"; content = "Reply with exactly: Hello from Flash Gordon." }
    )
    temperature = 0
    max_tokens = 512
    stream = $false
}

for ($i = 0; $i -lt $WarmupRuns; $i++) {
    $null = Invoke-JsonRequest -Body $hello
}

$decodeRates = @()
$prefillRates = @()
for ($i = 0; $i -lt $MeasuredRuns; $i++) {
    $response = Invoke-JsonRequest -Body $hello
    $completion = $response.Content | ConvertFrom-Json
    if ($completion.choices[0].message.content.Trim() -ne "Hello from Flash Gordon.") {
        throw "Identical-prompt response mismatch on measured run $($i + 1)."
    }
    $decodeRates += Get-HeaderDouble -Response $response -Name "X-Flash-Gordon-Decode-TPS"
    $prefillRates += Get-HeaderDouble -Response $response -Name "X-Flash-Gordon-Prefill-TPS"
}

$decodeMedian = Get-Median -Values $decodeRates
if ($decodeMedian -lt $MinimumDecodeTps) {
    throw ("Decode median {0:F6} tok/s is below required {1:F6} tok/s." -f
        $decodeMedian, $MinimumDecodeTps)
}

$tools = @(
    @{
        type = "function"
        function = @{
            name = "read_file"
            description = "Read a UTF-8 text file from the working directory."
            parameters = @{
                type = "object"
                properties = @{
                    path = @{ type = "string"; description = "Relative path to read." }
                }
                required = @("path")
                additionalProperties = $false
            }
        }
    }
)

$toolRequest = @{
    model = $Model
    messages = @(
        @{ role = "system"; content = "You are a coding agent. Use the provided tools when needed." },
        @{ role = "user"; content = "Read README.md and tell me the first Markdown heading." }
    )
    tools = $tools
    tool_choice = @{ type = "function"; function = @{ name = "read_file" } }
    temperature = 0
    max_tokens = 512
    stream = $false
}

$firstResponse = Invoke-JsonRequest -Body $toolRequest
$first = $firstResponse.Content | ConvertFrom-Json
$call = $first.choices[0].message.tool_calls[0]
if (-not $call -or $first.choices[0].finish_reason -ne "tool_calls" -or
    $call.function.name -ne "read_file") {
    throw "Model did not produce the required structured read_file tool call."
}
$arguments = $call.function.arguments | ConvertFrom-Json
if ($arguments.path -ne "README.md") {
    throw "Expected read_file path README.md, got $($arguments.path)."
}

$toolRequest.messages += @{
    role = "assistant"
    content = $null
    tool_calls = @($call)
}
$toolRequest.messages += @{
    role = "tool"
    tool_call_id = $call.id
    content = "# Flash Gordon`n`nFlash Gordon is a BC250 inference appliance."
}
$toolRequest.tool_choice = "none"
$continuationResponse = Invoke-JsonRequest -Body $toolRequest
$continuation = $continuationResponse.Content | ConvertFrom-Json
if ($continuation.choices[0].message.content -notmatch "# Flash Gordon") {
    throw "Model did not continue correctly after the tool result."
}

$streamRequest = @{
    model = $Model
    messages = @(
        @{ role = "system"; content = "You are a coding agent. Use the provided tools when needed." },
        @{ role = "user"; content = "Read README.md." }
    )
    tools = $tools
    tool_choice = @{ type = "function"; function = @{ name = "read_file" } }
    temperature = 0
    max_tokens = 512
    stream = $true
}
$stream = Invoke-JsonRequest -Body $streamRequest
$streamContent = ""
$streamCalls = @{}
$streamFinish = $null
$streamDone = $false
foreach ($line in ($stream.Content -split "\r?\n")) {
    if (-not $line.StartsWith("data: ")) {
        continue
    }
    $payload = $line.Substring(6)
    if ($payload -eq "[DONE]") {
        $streamDone = $true
        continue
    }
    $event = $payload | ConvertFrom-Json
    $choice = $event.choices[0]
    if ($null -ne $choice.delta.content) {
        $streamContent += [string]$choice.delta.content
    }
    foreach ($part in @($choice.delta.tool_calls)) {
        if ($null -eq $part) {
            continue
        }
        $index = [string]$part.index
        if (-not $streamCalls.ContainsKey($index)) {
            $streamCalls[$index] = @{ id = ""; type = ""; name = ""; arguments = "" }
        }
        if ($null -ne $part.id) {
            $streamCalls[$index].id += [string]$part.id
        }
        if ($null -ne $part.type) {
            $streamCalls[$index].type += [string]$part.type
        }
        if ($null -ne $part.function.name) {
            $streamCalls[$index].name += [string]$part.function.name
        }
        if ($null -ne $part.function.arguments) {
            $streamCalls[$index].arguments += [string]$part.function.arguments
        }
    }
    if ($null -ne $choice.finish_reason) {
        $streamFinish = [string]$choice.finish_reason
    }
}
if (-not $streamDone -or $streamFinish -ne "tool_calls" -or
    $streamCalls.Count -ne 1 -or -not $streamCalls["0"].id -or
    $streamCalls["0"].type -ne "function" -or
    $streamCalls["0"].name -ne "read_file" -or
    (($streamCalls["0"].arguments | ConvertFrom-Json).path -ne "README.md") -or
    $streamContent -match '<tool_call>|<function=|<parameter=') {
    throw "Streaming structured tool-call qualification failed."
}

[pscustomobject]@{
    Model = $Model
    MeasuredRuns = $MeasuredRuns
    DecodeRates = ($decodeRates | ForEach-Object { $_.ToString("F6", $InvariantCulture) })
    DecodeMedian = $decodeMedian.ToString("F6", $InvariantCulture)
    MinimumDecodeTps = $MinimumDecodeTps.ToString("F6", $InvariantCulture)
    PrefillRates = ($prefillRates | ForEach-Object { $_.ToString("F6", $InvariantCulture) })
    ToolCall = "$($call.function.name)($($call.function.arguments))"
    ToolContinuation = $continuation.choices[0].message.content.Trim()
    StreamingToolCall = "PASS"
    Result = "PASS"
}
