#requires -Version 7.0
<#
.SYNOPSIS
Measure Flash Gordon prefill, TTFT, decode, concurrency, and greedy text stability.
.DESCRIPTION
Queries an OpenAI-compatible API without changing the serving process. Streaming
responses expose terminal usage on current builds. Concurrent tests use that
generation's actual counts; the legacy single-stream suite documents paired
non-stream measurements used for engine rates.
.EXAMPLE
pwsh -NoProfile -File tools/hillclimb.ps1 -Quick -WriteReference
.EXAMPLE
pwsh -NoProfile -File tools/hillclimb.ps1
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:FG_BASE_URL) { $env:FG_BASE_URL.TrimEnd('/') } else { "http://192.168.42.42:8080/v1" }),
    [switch]$Quick,
    [switch]$WriteReference,
    [switch]$ConcurrentGeneration,
    [switch]$MixedPrefill,
    [switch]$CapacityProbe,
    [switch]$AllowSerializedBaseline,
    [string]$PromptSalt = '',
    [int]$ChatContextTokens = 64,
    [int]$AgentContextTokens = 32768,
    [int]$ConcurrentTokens = 256,
    [int]$ConcurrentRepeats = 3,
    [int]$ChatDelayMilliseconds = 200,
    [string]$ResultsDirectory = '',
    [int]$RequestTimeoutSeconds = 900
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')
$repo = Split-Path $PSScriptRoot -Parent
$resultRoot = Join-Path $repo 'results\hillclimb'
$runRoot = if ($ResultsDirectory) { $ResultsDirectory } else { Join-Path $resultRoot 'runs' }
$referencePath = Join-Path $resultRoot 'reference.json'
$null = New-Item -ItemType Directory -Path $runRoot -Force
$invariant = [Globalization.CultureInfo]::InvariantCulture
$handler = [Net.Http.SocketsHttpHandler]::new()
$handler.MaxConnectionsPerServer = 8
$client = [Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromSeconds($RequestTimeoutSeconds)
$null = $client.DefaultRequestHeaders.TryAddWithoutValidation('Accept', 'application/json, text/event-stream')
$runStarted = [DateTimeOffset]::Now
$records = [ordered]@{
    schema = 1
    metadata = [ordered]@{
        timestamp = $runStarted.ToString('o')
        base_url = $BaseUrl
        quick = [bool]$Quick
        model = $null
        ledger = $null
        git_sha = $null
        server_headers = [ordered]@{}
        notes = @(
            'SSE delta timestamps measure visible content events; the API does not expose token IDs or engine metric headers on streamed responses.',
            'ignore_eos/min_tokens are not implemented by src/api.c. Decode requests use greedy sampling and report actual usage token counts; max_tokens is only a ceiling.',
            'Long-context engine decode headers come from a cached follow-on non-stream continuation, not the exact SSE generation. This preserves the long KV context without a second cold prefill.'
        )
    }
    prefill = @()
    ttft = @()
    decode = @()
    concurrency = $null
    correctness = @()
    errors = @()
}
$script:prefillPrompts = @{}
$script:promptNonce = 0

function Get-HeaderMap {
    param([Net.Http.HttpResponseMessage]$Response)
    $map = [ordered]@{}
    foreach ($header in $Response.Headers) { $map[$header.Key] = (@($header.Value) -join ',') }
    foreach ($header in $Response.Content.Headers) { $map[$header.Key] = (@($header.Value) -join ',') }
    return $map
}

function Get-HeaderNumber {
    param([System.Collections.IDictionary]$Headers, [string]$Name)
    if (-not $Headers.Contains($Name) -or [string]::IsNullOrWhiteSpace([string]$Headers[$Name])) { return $null }
    $value = 0.0
    if ([double]::TryParse([string]$Headers[$Name], [Globalization.NumberStyles]::Float, $invariant, [ref]$value)) {
        return $value
    }
    return $null
}

function Get-Model {
    $response = $client.GetAsync("$BaseUrl/models").GetAwaiter().GetResult()
    try {
        $null = $response.EnsureSuccessStatusCode()
        $payload = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
        if (-not $payload.data -or @($payload.data).Count -lt 1 -or -not $payload.data[0].id) {
            throw 'GET /models returned no model id.'
        }
        return [string]$payload.data[0].id
    } finally { $response.Dispose() }
}

function New-Nonce {
    if ($PromptSalt) {
        $text = "$PromptSalt/$script:promptNonce"
        $script:promptNonce++
        return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
            [Text.Encoding]::UTF8.GetBytes($text))).Substring(0,32).ToLowerInvariant()
    }
    return [Guid]::NewGuid().ToString('N')
}

function New-Prompt {
    param([int]$TargetTokens, [string]$Purpose, [switch]$Agent)
    $nonce = New-Nonce
    $system = if ($Agent) {
        "Act as a coding agent. Run nonce $nonce. Be concise and follow the user's exact request."
    } else {
        "This is an isolated benchmark prompt with nonce $nonce. Answer briefly."
    }
    $content = if ($TargetTokens -le 0) {
        "/no_think nonce $nonce. $Purpose"
    } else {
        $builder = [Text.StringBuilder]::new([Math]::Min($TargetTokens * 8, 1500000))
        [void]$builder.Append("/no_think nonce $nonce. ")
        [void]$builder.Append((' hello' * [Math]::Max(1, $TargetTokens - 48)))
        [void]$builder.Append("`n$Purpose")
        $builder.ToString()
    }
    return [pscustomobject]@{
        Nonce = $nonce
        Messages = @(
            @{ role = 'system'; content = $system }
            @{ role = 'user'; content = $content }
        )
    }
}

function New-BodyJson {
    param([object[]]$Messages, [int]$MaxTokens, [bool]$Stream, [int]$Seed = 1729, [string]$User = '')
    $body = [ordered]@{
        model = $records.metadata.model
        messages = $Messages
        temperature = 0
        top_p = 1
        top_k = 1
        seed = $Seed
        max_tokens = $MaxTokens
        stream = $Stream
    }
    if ($User) { $body.user = $User }
    return ($body | ConvertTo-Json -Depth 30 -Compress)
}

function Start-Sse {
    param([object[]]$Messages, [int]$MaxTokens, [int]$Seed = 1729, [string]$User = '')
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$BaseUrl/chat/completions")
    $request.Content = [Net.Http.StringContent]::new((New-BodyJson $Messages $MaxTokens $true $Seed $User),
        [Text.Encoding]::UTF8, 'application/json')
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    if (-not $response.IsSuccessStatusCode) {
        $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $status = [int]$response.StatusCode
        $response.Dispose()
        $request.Dispose()
        throw "SSE request failed HTTP ${status}: $text"
    }
    $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
    return [pscustomobject]@{
        Request = $request
        Response = $response
        Reader = $reader
        Watch = $watch
        Headers = (Get-HeaderMap $response)
        TimesMs = [Collections.Generic.List[double]]::new()
        Text = [Text.StringBuilder]::new()
        Done = $false
        FirstTokenMs = $null
        Usage = $null
        Id = $null
        Prefix = $null
        FirstContentTimestamp = $null
        LastContentTimestamp = $null
        CompletedTimestamp = $null
    }
}

function Consume-SseLine {
    param($Session, [string]$Line)
    if (-not $Line.StartsWith('data: ')) { return }
    $data = $Line.Substring(6)
    if ($data -eq '[DONE]') { $Session.Done = $true; return }
    try { $event = $data | ConvertFrom-Json } catch { return }
    if ($event.PSObject.Properties['id']) { $Session.Id = [string]$event.id }
    if ($event.PSObject.Properties['error']) { throw "SSE error: $($event.error.message)" }
    if ($event.PSObject.Properties['usage']) { $Session.Usage = $event.usage }
    if ($event.PSObject.Properties['flash_gordon']) { $Session.Prefix = $event.flash_gordon }
    if (-not $event.choices -or @($event.choices).Count -eq 0) { return }
    $delta = $event.choices[0].delta
    $part = ''
    if ($delta -and $delta.PSObject.Properties['content'] -and $null -ne $delta.content) {
        $part = [string]$delta.content
    } elseif ($delta -and $delta.PSObject.Properties['reasoning_content'] -and $null -ne $delta.reasoning_content) {
        $part = [string]$delta.reasoning_content
    }
    if ($part.Length -gt 0) {
        $when = $Session.Watch.Elapsed.TotalMilliseconds
        if ($null -eq $Session.FirstTokenMs) {
            $Session.FirstTokenMs = $when
            $Session.FirstContentTimestamp = [Diagnostics.Stopwatch]::GetTimestamp()
        }
        $Session.LastContentTimestamp = [Diagnostics.Stopwatch]::GetTimestamp()
        $Session.TimesMs.Add($when)
        [void]$Session.Text.Append($part)
    }
}

function Finish-Sse {
    param($Session)
    try {
        while (-not $Session.Done) {
            $line = $Session.Reader.ReadLineAsync().GetAwaiter().GetResult()
            if ($null -eq $line) { break }
            Consume-SseLine $Session $line
        }
        $Session.Watch.Stop()
        $Session.CompletedTimestamp = [Diagnostics.Stopwatch]::GetTimestamp()
        return $Session
    } finally {
        $Session.Reader.Dispose()
        $Session.Response.Dispose()
        $Session.Request.Dispose()
    }
}

function Invoke-NonStream {
    param([object[]]$Messages, [int]$MaxTokens, [int]$Seed = 1729, [string]$User = '')
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$BaseUrl/chat/completions")
    $request.Content = [Net.Http.StringContent]::new((New-BodyJson $Messages $MaxTokens $false $Seed $User),
        [Text.Encoding]::UTF8, 'application/json')
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = $client.SendAsync($request).GetAwaiter().GetResult()
    try {
        $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $watch.Stop()
        if (-not $response.IsSuccessStatusCode) { throw "HTTP $([int]$response.StatusCode): $text" }
        $payload = $text | ConvertFrom-Json
        return [pscustomobject]@{
            Payload = $payload
            Headers = (Get-HeaderMap $response)
            WallSeconds = $watch.Elapsed.TotalSeconds
            PromptTokens = [int]$payload.usage.prompt_tokens
            CompletionTokens = [int]$payload.usage.completion_tokens
            Content = [string]$payload.choices[0].message.content
            FinishReason = [string]$payload.choices[0].finish_reason
        }
    } finally {
        $response.Dispose()
        $request.Dispose()
    }
}

function Get-Percentile {
    param([double[]]$Values, [double]$Quantile)
    if (-not $Values -or $Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $index = [Math]::Min($sorted.Count - 1, [Math]::Max(0, [int][Math]::Ceiling($Quantile * $sorted.Count) - 1))
    return [double]$sorted[$index]
}

function Get-SseStats {
    param($Session)
    $times = @($Session.TimesMs)
    $gaps = @()
    for ($i = 1; $i -lt $times.Count; $i++) { $gaps += ([double]$times[$i] - [double]$times[$i - 1]) }
    $duration = if ($times.Count -gt 1) { ($times[-1] - $times[0]) / 1000.0 } else { 0.0 }
    $intervalTps = if ($times.Count -gt 1 -and $duration -gt 0) { ($times.Count - 1) / $duration } else { 0.0 }
    return [ordered]@{
        first_content_ms = $Session.FirstTokenMs
        wall_seconds = $Session.Watch.Elapsed.TotalSeconds
        visible_delta_events = $times.Count
        sse_inter_event_tps_excluding_first = $intervalTps
        inter_event_p50_ms = Get-Percentile $gaps 0.50
        inter_event_p90_ms = Get-Percentile $gaps 0.90
        finish_received = [bool]$Session.Done
        visible_text = $Session.Text.ToString()
    }
}

function New-FreshBenchmarkPair {
    param([int]$TargetTokens, [string]$Name)
    $purpose = "Return a short plain-text response. Benchmark case $Name."
    $streamPrompt = New-Prompt $TargetTokens $purpose
    $enginePrompt = New-Prompt $TargetTokens $purpose
    $full = Invoke-NonStream $enginePrompt.Messages 1
    $stream = Finish-Sse (Start-Sse $streamPrompt.Messages 1)
    $sse = Get-SseStats $stream
    $streamPrompt | Add-Member -NotePropertyName InitialAssistant -NotePropertyValue $sse.visible_text
    $streamPrompt | Add-Member -NotePropertyName StreamWallSeconds -NotePropertyValue $sse.wall_seconds
    $script:prefillPrompts[$Name] = $streamPrompt
    $cacheHeader = [string]$full.Headers['X-Flash-Gordon-Prefix-Cache']
    $reused = Get-HeaderNumber $full.Headers 'X-Flash-Gordon-Reused-Tokens'
    return [pscustomobject]@{
        name = $Name
        target_prompt_tokens = $TargetTokens
        engine_prompt_nonce = $enginePrompt.Nonce
        stream_prompt_nonce = $streamPrompt.Nonce
        stream_first_content_ms = $sse.first_content_ms
        stream_wall_seconds = $sse.wall_seconds
        stream_visible_delta_events = $sse.visible_delta_events
        usage_prompt_tokens = $full.PromptTokens
        stream_usage_prompt_tokens = $null
        client_prefill_tps = if ($null -ne $sse.first_content_ms -and $sse.first_content_ms -gt 0) {
            $full.PromptTokens / ($sse.first_content_ms / 1000.0)
        } else { $null }
        engine_prefill_tps = Get-HeaderNumber $full.Headers 'X-Flash-Gordon-Prefill-TPS'
        engine_prefill_seconds = Get-HeaderNumber $full.Headers 'X-Flash-Gordon-Prefill-Seconds'
        engine_prefilled_tokens = Get-HeaderNumber $full.Headers 'X-Flash-Gordon-Prefilled-Tokens'
        engine_reused_tokens = $reused
        prefix_cache = $cacheHeader
        cold_measurement = ($cacheHeader -eq 'miss' -and ($null -eq $reused -or $reused -eq 0))
        stream_completion_usage = $null
        usage_source = 'same-size independently salted non-stream prompt; stream endpoint omits usage'
        nonstream_wall_seconds = $full.WallSeconds
    }
}

function Invoke-Ttft {
    param([string]$Name, [object[]]$Messages, [int]$MaxTokens = 32)
    $session = Finish-Sse (Start-Sse $Messages $MaxTokens)
    $stats = Get-SseStats $session
    $headers = $session.Headers
    return [pscustomobject]@{
        name = $Name
        ttft_ms = $stats.first_content_ms
        total_wall_seconds = $stats.wall_seconds
        visible_delta_events = $stats.visible_delta_events
        visible_text = $stats.visible_text
        finish_received = $stats.finish_received
        ledger = [string]$headers['X-Flash-Gordon-Ledger']
        streamed_headers_include_engine_metrics = ($headers.Contains('X-Flash-Gordon-Prefill-TPS'))
    }
}

function Invoke-Decode {
    param(
        [string]$Name,
        [int]$ContextTokens,
        [int]$MaxTokens,
        [object[]]$Messages,
        [string]$InitialAssistant = '',
        [switch]$UseCachedContinuation
    )
    $followMessages = $null
    if ($UseCachedContinuation) {
        $streamMessages = @($Messages) + @(
            @{ role = 'assistant'; content = $InitialAssistant }
            @{ role = 'user'; content = '/no_think Write a long comma-separated list of distinct concise items. Continue until the token limit.' }
        )
    } else {
        $streamMessages = $Messages
    }
    $stream = Finish-Sse (Start-Sse $streamMessages $MaxTokens)
    $sse = Get-SseStats $stream
    if ($UseCachedContinuation) {
        $followMessages = @($streamMessages) + @(
            @{ role = 'assistant'; content = $sse.visible_text }
            @{ role = 'user'; content = '/no_think Continue the list with new items until the token limit.' }
        )
        $nonstream = Invoke-NonStream $followMessages $MaxTokens
    } else {
        $nonstream = Invoke-NonStream $Messages $MaxTokens
    }
    return [pscustomobject]@{
        name = $Name
        target_context_tokens = $ContextTokens
        usage_prompt_tokens = $nonstream.PromptTokens
        requested_max_tokens = $MaxTokens
        actual_completion_tokens = $null
        engine_actual_completion_tokens = $nonstream.CompletionTokens
        engine_finish_reason = $nonstream.FinishReason
        stream_visible_delta_events = $sse.visible_delta_events
        client_sse_inter_event_tps_excluding_first = $sse.sse_inter_event_tps_excluding_first
        inter_event_p50_ms = $sse.inter_event_p50_ms
        inter_event_p90_ms = $sse.inter_event_p90_ms
        engine_decode_tps = Get-HeaderNumber $nonstream.Headers 'X-Flash-Gordon-Decode-TPS'
        engine_decode_seconds = Get-HeaderNumber $nonstream.Headers 'X-Flash-Gordon-Decode-Seconds'
        engine_prefix_cache = [string]$nonstream.Headers['X-Flash-Gordon-Prefix-Cache']
        engine_reused_tokens = Get-HeaderNumber $nonstream.Headers 'X-Flash-Gordon-Reused-Tokens'
        engine_context_tokens = Get-HeaderNumber $nonstream.Headers 'X-Flash-Gordon-Context-Tokens'
        sse_wall_seconds = $sse.wall_seconds
        engine_response_wall_seconds = $nonstream.WallSeconds
        stream_actual_tokens = $null
        engine_metrics_same_stream_request = $false
        engine_metrics_source = if ($UseCachedContinuation) {
            'cached follow-on non-stream continuation; same long context, different continuation text'
        } else {
            'non-stream duplicate request; stream endpoint omits X-Flash-Gordon metrics'
        }
        ignore_eos_supported = $false
        min_tokens_supported = $false
    }
}

function Invoke-Correctness {
    $prompts = @(
        @{ name = 'arithmetic'; prompt = 'What is 17 plus 25? Output 64 copies of the answer, 42, separated by spaces; no other text.' },
        @{ name = 'capital'; prompt = 'What is the capital city of Japan? Output 64 copies of the answer, Tokyo, separated by spaces; no other text.' },
        @{ name = 'sequence'; prompt = 'Continue 2, 4, 6, 8. Output 32 repetitions of "10, 12", separated by spaces; no other text.' },
        @{ name = 'spelling'; prompt = 'Spell "accommodation". Output 64 copies of the correct spelling separated by spaces; no other text.' },
        @{ name = 'logic'; prompt = 'Which is heavier: one kilogram of steel or one kilogram of feathers? Output 64 copies of "They weigh the same." separated by spaces; no other text.' }
    )
    $refs = @{}
    if (Test-Path -LiteralPath $referencePath) {
        $reference = Get-Content -Raw -LiteralPath $referencePath | ConvertFrom-Json
        foreach ($item in $reference.cases) { $refs[[string]$item.name] = $item }
    }
    $current = [Collections.Generic.List[object]]::new()
    foreach ($case in $prompts) {
        $messages = @(
            @{ role = 'system'; content = 'Answer the user directly and exactly as requested.' }
            @{ role = 'user'; content = "/no_think $($case.prompt)" }
        )
        $result = Invoke-NonStream $messages 64
        $text = $result.Content.Trim()
        $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($text))).ToLowerInvariant()
        $match = $null -ne $refs[$case.name] -and [string]$refs[$case.name].sha256 -eq $hash
        $row = [pscustomobject]@{
            name = $case.name
            prompt = $case.prompt
            text = $text
            sha256 = $hash
            reference_match = if ($WriteReference) { $null } else { $match }
            actual_tokens = $result.CompletionTokens
            requested_max_tokens = 64
            finish_reason = $result.FinishReason
        }
        $current.Add($row)
    }
    if ($WriteReference) {
        $reference = [ordered]@{
            schema = 1
            model = $records.metadata.model
            greedy = $true
            max_tokens = 64
            created_at = [DateTimeOffset]::Now.ToString('o')
            cases = @($current | ForEach-Object {
                [ordered]@{ name = $_.name; prompt = $_.prompt; text = $_.text; sha256 = $_.sha256; actual_tokens = $_.actual_tokens }
            })
            limitation = 'Text SHA-256 only; this API does not expose generated token IDs.'
        }
        $reference | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $referencePath -Encoding utf8
        $records.metadata.reference_file = 'results/hillclimb/reference.json'
    } elseif (-not (Test-Path -LiteralPath $referencePath)) {
        throw "Reference file is missing. Run once with -WriteReference against the intended production model."
    }
    return @($current)
}

function Get-GenerationStats {
    param($Session, [string]$Name)
    $sse = Get-SseStats $Session
    $gaps = @(
        for ($i = 1; $i -lt $Session.TimesMs.Count; $i++) {
            $Session.TimesMs[$i] - $Session.TimesMs[$i - 1]
        }
    )
    $tokens = if ($null -ne $Session.Usage) { [int]$Session.Usage.completion_tokens } else { $null }
    $decodeSeconds = if ($null -ne $Session.FirstTokenMs) {
        $Session.Watch.Elapsed.TotalSeconds - $Session.FirstTokenMs / 1000.0
    } else { 0.0 }
    $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
        [Text.Encoding]::UTF8.GetBytes($sse.visible_text))).ToLowerInvariant()
    return [pscustomobject]@{
        name = $Name
        request_id = $Session.Id
        ttft_ms = $sse.first_content_ms
        wall_seconds = $sse.wall_seconds
        completion_tokens = $tokens
        prompt_tokens = if ($Session.Usage) { [int]$Session.Usage.prompt_tokens } else { $null }
        prefix = $Session.Prefix
        first_content_clock_ticks = $Session.FirstContentTimestamp
        last_content_clock_ticks = $Session.LastContentTimestamp
        client_decode_tps = if ($null -ne $tokens -and $decodeSeconds -gt 0) { $tokens / $decodeSeconds } else { $null }
        client_end_to_end_tps = if ($null -ne $tokens -and $sse.wall_seconds -gt 0) { $tokens / $sse.wall_seconds } else { $null }
        visible_delta_events = $sse.visible_delta_events
        inter_event_p50_ms = Get-Percentile $gaps 0.50
        inter_event_p95_ms = Get-Percentile $gaps 0.95
        inter_event_max_ms = ($gaps | Measure-Object -Maximum).Maximum
        visible_event_times_ms = $Session.TimesMs.ToArray()
        finish_received = $sse.finish_received
        sha256 = $hash
        visible_text = $sse.visible_text
    }
}

function New-ConcurrentSse {
    param([object[]]$Messages, [int]$MaxTokens, [string]$User)
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$BaseUrl/chat/completions")
    $request.Content = [Net.Http.StringContent]::new((New-BodyJson $Messages $MaxTokens $true 1729 $User),
        [Text.Encoding]::UTF8, 'application/json')
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $task = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
    return [pscustomobject]@{
        Request = $request; Response = $null; Reader = $null; Watch = $watch
        Task = $task; AwaitingHeaders = $true; Headers = @{}
        TimesMs = [Collections.Generic.List[double]]::new()
        Text = [Text.StringBuilder]::new(); Done = $false; FirstTokenMs = $null
        Usage = $null; Id = $null; Prefix = $null; FirstContentTimestamp = $null
        LastContentTimestamp = $null; CompletedTimestamp = $null
    }
}

function Invoke-Concurrency {
    param([object[]]$ChatMessages, [object[]]$AgentMessages, [int]$MaxTokens = 32, [int]$ChatDelayMs = 0)
    $sessions = @()
    $batchStarted = [Diagnostics.Stopwatch]::GetTimestamp()
    try {
        $agent = New-ConcurrentSse $AgentMessages $MaxTokens 'hillclimb-agent'
        $sessions += $agent
        if ($ChatDelayMs -gt 0) { Start-Sleep -Milliseconds $ChatDelayMs }
        $chat = New-ConcurrentSse $ChatMessages $MaxTokens 'hillclimb-chat'
        $sessions += $chat
        $active = @($sessions)
        while ($active.Count -gt 0) {
            $tasks = [Threading.Tasks.Task[]]@($active | ForEach-Object { $_.Task })
            $index = [Threading.Tasks.Task]::WaitAny($tasks, $RequestTimeoutSeconds * 1000)
            if ($index -lt 0) { throw 'Concurrent stream read timed out.' }
            $session = $active[$index]
            if ($session.AwaitingHeaders) {
                $session.Response = $session.Task.GetAwaiter().GetResult()
                if (-not $session.Response.IsSuccessStatusCode) {
                    throw "Concurrent HTTP $([int]$session.Response.StatusCode): $($session.Response.Content.ReadAsStringAsync().GetAwaiter().GetResult())"
                }
                $session.Headers = Get-HeaderMap $session.Response
                $stream = $session.Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
                $session.Reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
                $session.AwaitingHeaders = $false
            } else {
                $line = $session.Task.GetAwaiter().GetResult()
                if ($null -eq $line) { throw 'Concurrent SSE closed without [DONE].' }
                Consume-SseLine $session $line
            }
            if ($session.Done) {
                $session.Watch.Stop()
                $session.CompletedTimestamp = [Diagnostics.Stopwatch]::GetTimestamp()
                $active = @($active | Where-Object { -not [object]::ReferenceEquals($_, $session) })
            } else { $session.Task = $session.Reader.ReadLineAsync() }
        }
        $rows = @((Get-GenerationStats $chat 'chat-style'), (Get-GenerationStats $agent 'agent-style'))
        $usageAvailable = $null -ne $chat.Usage -and $null -ne $agent.Usage
        $first = [Math]::Min([double]$chat.FirstContentTimestamp, [double]$agent.FirstContentTimestamp)
        $last = [Math]::Max([double]$chat.CompletedTimestamp, [double]$agent.CompletedTimestamp)
        $overlap = ([Math]::Min([double]$chat.LastContentTimestamp, [double]$agent.LastContentTimestamp) -
            [Math]::Max([double]$chat.FirstContentTimestamp, [double]$agent.FirstContentTimestamp)) /
            [Diagnostics.Stopwatch]::Frequency
        $aggregate = if ($usageAvailable -and $last -gt $first) {
            ($chat.Usage.completion_tokens + $agent.Usage.completion_tokens) /
                (($last - $first) / [Diagnostics.Stopwatch]::Frequency)
        } else { $null }
        return [pscustomobject]@{
            streams = $rows
            aggregate_client_decode_tps = $aggregate
            aggregate_window_seconds = ($last - $first) / [Diagnostics.Stopwatch]::Frequency
            batch_wall_seconds = ($last - $batchStarted) / [Diagnostics.Stopwatch]::Frequency
            aggregate_wall_tps = if ($usageAvailable) {
                ($chat.Usage.completion_tokens + $agent.Usage.completion_tokens) /
                    (($last - $batchStarted) / [Diagnostics.Stopwatch]::Frequency)
            } else { $null }
            overlapping_visible_decode_seconds = [Math]::Max(0.0, [double]$overlap)
            chat_first_token_before_agent_first_token = $chat.FirstContentTimestamp -lt $agent.FirstContentTimestamp
            token_count_source = if ($usageAvailable) { 'usage on these exact SSE streams' } else { 'unavailable; do not substitute delta events' }
            denominator = 'sum actual completion tokens / (latest DONE - earliest first visible token), including gaps and fill/drain'
        }
    } finally {
        foreach ($session in $sessions) {
            if ($session.Reader) { $session.Reader.Dispose() }
            if ($session.Response) { $session.Response.Dispose() }
            $session.Request.Dispose()
        }
    }
}

function Invoke-ConcurrentGeneration {
    if ($ConcurrentTokens -lt 2 -or $ConcurrentRepeats -lt 1 -or $ChatContextTokens -lt 0 -or
        $AgentContextTokens -lt 0 -or $ChatDelayMilliseconds -lt 0) { throw 'Invalid concurrency parameters.' }
    if ($CapacityProbe -and -not $MixedPrefill) { throw 'CapacityProbe requires MixedPrefill.' }
    $trials = @()
    $pendingAccounting = $false
    $purpose = 'Print a long numbered list of distinct practical software tasks, one per line. Continue until the token limit.'
    for ($repeat = 0; $repeat -lt $ConcurrentRepeats; $repeat++) {
        $chat = New-Prompt $ChatContextTokens 'Output exactly Ready.'
        $agent = New-Prompt $AgentContextTokens $(if ($MixedPrefill) { $purpose } else { 'Output exactly Ready.' }) -Agent
        $chatPrime = Invoke-NonStream $chat.Messages 32 -User 'hillclimb-chat'
        $chatMessages = @($chat.Messages) + @(
            @{ role = 'assistant'; content = $chatPrime.Content }
            @{ role = 'user'; content = "/no_think $purpose" }
        )
        $agentMessages = $agent.Messages
        if (-not $MixedPrefill) {
            $agentPrime = Invoke-NonStream $agent.Messages 32 -User 'hillclimb-agent'
            $agentMessages = @($agent.Messages) + @(
                @{ role = 'assistant'; content = $agentPrime.Content }
                @{ role = 'user'; content = "/no_think $purpose" }
            )
        }
        $chatIsolated = Finish-Sse (Start-Sse $chatMessages $ConcurrentTokens -User 'hillclimb-chat')
        $agentIsolated = if (-not $CapacityProbe) {
            Finish-Sse (Start-Sse $agentMessages $ConcurrentTokens -User 'hillclimb-agent')
        } else { $null }
        $baseline = @((Get-GenerationStats $chatIsolated 'chat-style'),
            $(if ($agentIsolated) { Get-GenerationStats $agentIsolated 'agent-style' } else { $null }))
        $null = Invoke-NonStream $chat.Messages 32 -User 'hillclimb-chat'
        if (-not $MixedPrefill) { $null = Invoke-NonStream $agent.Messages 32 -User 'hillclimb-agent' }
        $load = Invoke-Concurrency $chatMessages $agentMessages $ConcurrentTokens `
            -ChatDelayMs $(if ($MixedPrefill) { $ChatDelayMilliseconds } else { 0 })
        $trials += [pscustomobject]@{ repeat = $repeat; isolated = $baseline; load = $load }
        $records.concurrency = [pscustomobject]@{
            mode = if ($MixedPrefill) { 'chat-decode-during-agent-cold-prefill' } else { 'two-warm-prefix-generations' }
            capacity_probe = [bool]$CapacityProbe; complete = $false; trials = $trials
        }
        if ($AllowSerializedBaseline -and @($load.streams | Where-Object { $null -eq $_.completion_tokens }).Count) {
            $pendingAccounting = $true
            foreach ($index in 0,1) {
                $load.streams[$index] | Add-Member same_prompt_hash_match (
                    $(if ($baseline[$index]) {
                        $load.streams[$index].sha256 -eq $baseline[$index].sha256
                    } else { $null }))
                $load.streams[$index] | Add-Member requested_token_ceiling $ConcurrentTokens
            }
            Write-Host 'BASELINE requires exact request-ID-matched server log usage; no event-count substitution.'
            continue
        }
        foreach ($index in 0,1) {
            $row = $load.streams[$index]
            if ($null -eq $row.completion_tokens) { throw 'ConcurrentGeneration requires SSE usage from this branch.' }
            if ($row.completion_tokens -lt 32) { throw "Insufficient sustained generation: $($row.name) emitted $($row.completion_tokens) tokens." }
            $row | Add-Member isolated_decode_tps $(if ($baseline[$index]) { $baseline[$index].client_decode_tps } else { $null })
            $row | Add-Member decode_retention $(if ($baseline[$index]) { $row.client_decode_tps / $baseline[$index].client_decode_tps } else { $null })
            $row | Add-Member isolated_ttft_ms $(if ($baseline[$index]) { $baseline[$index].ttft_ms } else { $null })
            $row | Add-Member same_prompt_hash_match $(if ($baseline[$index]) { $row.sha256 -eq $baseline[$index].sha256 } else { $null })
            $row | Add-Member requested_token_ceiling $ConcurrentTokens
        }
        $load | Add-Member aggregate_speedup_over_mean_isolated $(if (-not $CapacityProbe) {
            $load.aggregate_client_decode_tps / (($baseline[0].client_decode_tps + $baseline[1].client_decode_tps) / 2.0)
        } else { $null })
        if (-not $AllowSerializedBaseline -and
            ($null -eq $load.streams[0].prefix -or $load.streams[0].prefix.reused_tokens -le 0)) {
            throw 'Chat continuation did not reuse its primed session.'
        }
        if ($MixedPrefill -and $load.streams[1].prefix.reused_tokens -ne 0) {
            throw 'Agent cold prefill unexpectedly reused a prefix.'
        }
        if (-not $AllowSerializedBaseline -and -not $MixedPrefill -and $load.streams[1].prefix.reused_tokens -le 0) {
            throw 'Agent continuation did not reuse its primed session.'
        }
        if (-not $AllowSerializedBaseline -and -not $MixedPrefill -and $load.overlapping_visible_decode_seconds -le 0) {
            throw 'The requests did not overlap in generation; this is not a pipeline throughput sample.'
        }
        if ($CapacityProbe) {
            $residentChat = New-Prompt 0 'Output exactly OK.'
            $residentReply = Invoke-NonStream $residentChat.Messages 16 -User 'hillclimb-chat'
            if ($residentReply.CompletionTokens -lt 1) { throw 'Chat failed with the near-limit agent resident.' }
            $load | Add-Member post_near_limit_chat ([pscustomobject]@{
                user = 'hillclimb-chat'; completion_tokens = $residentReply.CompletionTokens
                prompt_tokens = $residentReply.PromptTokens; wall_seconds = $residentReply.WallSeconds
                text = $residentReply.Content; headers = $residentReply.Headers
                note = 'Fresh chat resets only chat affinity after the long request completes; agent affinity is not reassigned.'
            })
        }
        Write-Host ("CONCURRENT repeat={0} mixed={1} aggregate_tps={2:F2} chat_ttft_ms={3:F1} chat_tps={4:F2} agent_tps={5:F2}" -f
            $repeat, [bool]$MixedPrefill, $load.aggregate_client_decode_tps, $load.streams[0].ttft_ms,
            $load.streams[0].client_decode_tps, $load.streams[1].client_decode_tps)
    }
    return [pscustomobject]@{
        mode = if ($MixedPrefill) { 'chat-decode-during-agent-cold-prefill' } else { 'two-warm-prefix-generations' }
        capacity_probe = [bool]$CapacityProbe
        serialized_baseline_allowed = [bool]$AllowSerializedBaseline
        prompt_salt = $PromptSalt
        complete = -not $pendingAccounting
        requires_server_log_accounting = $pendingAccounting
        trials = $trials
        chat_ttft_p50_ms = Get-Percentile @($trials | ForEach-Object { $_.load.streams[0].ttft_ms }) 0.5
        chat_ttft_p95_ms = Get-Percentile @($trials | ForEach-Object { $_.load.streams[0].ttft_ms }) 0.95
        aggregate_tps_median = if ($pendingAccounting) { $null } else {
            Get-Percentile @($trials | ForEach-Object { $_.load.aggregate_client_decode_tps }) 0.5
        }
    }
}

try {
    $records.metadata.model = Get-Model
    $records.metadata.run_id = [Guid]::NewGuid().ToString('N')
    $records.metadata.reference_write_requested = [bool]$WriteReference
    $modelHeadersPrompt = New-Prompt 0 'Return exactly the word ready.'
    $headerProbe = Invoke-NonStream $modelHeadersPrompt.Messages 1
    $records.metadata.server_headers = $headerProbe.Headers
    $records.metadata.ledger = [string]$headerProbe.Headers['X-Flash-Gordon-Ledger']
    foreach ($name in @('X-Flash-Gordon-Git-SHA', 'X-Git-SHA', 'X-Flash-Gordon-Git-Commit')) {
        if ($headerProbe.Headers.Contains($name)) { $records.metadata.git_sha = $headerProbe.Headers[$name]; break }
    }
    if ($ConcurrentGeneration) {
        $records.concurrency = Invoke-ConcurrentGeneration
        $records.metadata.correctness_pass = @($records.concurrency.trials | Where-Object {
            -not $_.load.streams[0].same_prompt_hash_match -or
                (-not $CapacityProbe -and -not $_.load.streams[1].same_prompt_hash_match)
        }).Count -eq 0
    } else {
    $records.metadata.capacity_headers = @($headerProbe.Headers.Keys | Where-Object { $_ -match '(?i)capacity|concurr|slot' } | ForEach-Object { "$_=$($headerProbe.Headers[$_])" })

    if ($Quick) {
        $prefill = New-FreshBenchmarkPair 4096 'prefill-4k'
        $records.prefill += $prefill
        $records.ttft += [pscustomobject]@{
            name = '4k-cold'; ttft_ms = $prefill.stream_first_content_ms
            total_wall_seconds = $prefill.stream_wall_seconds
            prompt_tokens = $prefill.usage_prompt_tokens
            finish_received = $true
            measurement_source = 'same cold unique-prompt stream as prefill-4k'
        }
        $shortPrompt = New-Prompt 0 'Write a comma-separated list of distinct short words; continue until the token limit.'
        $records.decode += Invoke-Decode 'short-context' 64 32 $shortPrompt.Messages
    } else {
        $prefill4 = New-FreshBenchmarkPair 4096 'prefill-4k'
        $records.prefill += $prefill4
        $records.ttft += [pscustomobject]@{
            name = '4k-cold'; ttft_ms = $prefill4.stream_first_content_ms
            total_wall_seconds = $prefill4.stream_wall_seconds
            prompt_tokens = $prefill4.usage_prompt_tokens
            finish_received = $true
            measurement_source = 'same cold unique-prompt stream as prefill-4k'
        }
        $records.prefill += New-FreshBenchmarkPair 32768 'prefill-32k'
        $prefill32Prompt = $script:prefillPrompts['prefill-32k']
        $decode32 = Invoke-Decode 'context-32k' 32768 256 $prefill32Prompt.Messages `
            -InitialAssistant $prefill32Prompt.InitialAssistant -UseCachedContinuation
        $records.decode += $decode32
        $records.prefill += New-FreshBenchmarkPair 131072 'prefill-128k'
        $prefill128Prompt = $script:prefillPrompts['prefill-128k']
        $records.decode += Invoke-Decode 'context-128k' 131072 256 $prefill128Prompt.Messages `
            -InitialAssistant $prefill128Prompt.InitialAssistant -UseCachedContinuation
        $records.decode += Invoke-Decode 'short-context' 64 256 (New-Prompt 64 'Generate a long comma-separated list of distinct words.').Messages
    }

    $chatPrompt = New-Prompt 0 'Say hello in one sentence and do not explain.'
    $chatTtft = Invoke-Ttft 'chat-30-token-ceiling' $chatPrompt.Messages 30
    $chatUsage = Invoke-NonStream $chatPrompt.Messages 30
    $chatTtft | Add-Member -NotePropertyName prompt_tokens -NotePropertyValue $chatUsage.PromptTokens
    $chatTtft | Add-Member -NotePropertyName usage_completion_tokens -NotePropertyValue $chatUsage.CompletionTokens
    $records.ttft += $chatTtft

    $prefixPrompt = New-Prompt 3900 'Read this project context and answer the final question briefly.' -Agent
    $prime = Invoke-NonStream $prefixPrompt.Messages 32
    $continuationMessages = @($prefixPrompt.Messages) + @(
        @{ role = 'assistant'; content = $prime.Content }
        @{ role = 'user'; content = 'Using the context above, state the next practical step in one sentence.' }
    )
    $cachePrimer = Invoke-NonStream $continuationMessages 16
    $cachedMessages = @($continuationMessages) + @(
        @{ role = 'assistant'; content = $cachePrimer.Content }
        @{ role = 'user'; content = 'Given the same context, give the next action in one brief sentence.' }
    )
    $cachedTtft = Invoke-Ttft 'agent-style-cached-3.9k-prefix' $cachedMessages 32
    $cachedTtft | Add-Member -NotePropertyName prompt_tokens -NotePropertyValue $cachePrimer.PromptTokens
    $cachedTtft | Add-Member -NotePropertyName engine_reused_tokens -NotePropertyValue (Get-HeaderNumber $cachePrimer.Headers 'X-Flash-Gordon-Reused-Tokens')
    $cachedTtft | Add-Member -NotePropertyName engine_prefix_cache -NotePropertyValue ([string]$cachePrimer.Headers['X-Flash-Gordon-Prefix-Cache'])
    $records.ttft += $cachedTtft

    if (-not $Quick) {
        $chatConcurrent = New-Prompt 0 'You are ChatGPT-style. Answer with a friendly greeting.'
        $agentConcurrentMessages = @($cachedMessages) + @(
            @{ role = 'assistant'; content = $cachedTtft.visible_text }
            @{ role = 'user'; content = '/no_think As the coding agent, inspect the existing context and state the next action.' }
        )
        $records.concurrency = Invoke-Concurrency $chatConcurrent.Messages $agentConcurrentMessages
    }

    $records.correctness = @(Invoke-Correctness)
    $failedCorrectness = @($records.correctness | Where-Object { $_.reference_match -eq $false })
    $records.metadata.correctness_pass = ($WriteReference -or $failedCorrectness.Count -eq 0)
    }
    $records.metadata.elapsed_seconds = ([DateTimeOffset]::Now - $runStarted).TotalSeconds
    $stamp = $runStarted.ToLocalTime().ToString('yyyyMMdd-HHmmss')
    $runPath = Join-Path $runRoot "$stamp.json"
    $records | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $runPath -Encoding utf8
    Write-Host "HILLCLIMB result=$runPath model=$($records.metadata.model) quick=$([bool]$Quick)"
    Write-Host ("HILLCLIMB elapsed_s={0:F1} prefill={1} decode={2} ttft={3} correctness_pass={4}" -f
        $records.metadata.elapsed_seconds, @($records.prefill).Count, @($records.decode).Count,
        @($records.ttft).Count, $records.metadata.correctness_pass)
    if (-not $records.metadata.correctness_pass -or
        ($ConcurrentGeneration -and -not $records.concurrency.complete)) { exit 2 }
} catch {
    $records.errors += [pscustomobject]@{ message = $_.Exception.Message; script_stack = $_.ScriptStackTrace }
    $records.metadata.elapsed_seconds = ([DateTimeOffset]::Now - $runStarted).TotalSeconds
    $stamp = $runStarted.ToLocalTime().ToString('yyyyMMdd-HHmmss')
    $runPath = Join-Path $runRoot "$stamp.json"
    $records | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $runPath -Encoding utf8
    Write-Error "Benchmark failed; partial results saved to $runPath. $($_.Exception.Message)"
    exit 1
} finally {
    $client.Dispose()
}
