function Get-ABHash {
    param([string]$Text)
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
        [Text.Encoding]::UTF8.GetBytes($Text))).ToLowerInvariant()
}

function Get-ABMedian {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    $n = $sorted.Count
    if (-not $n) { throw 'No observations.' }
    if ($n % 2) { return $sorted[[int]($n / 2 - 0.5)] }
    return ($sorted[$n / 2 - 1] + $sorted[$n / 2]) / 2
}

function Get-ABQuantile {
    param([double[]]$Values, [double]$Quantile)
    $sorted = @($Values | Sort-Object)
    $index = [Math]::Min($sorted.Count - 1, [Math]::Max(0, [int][Math]::Ceiling($Quantile * $sorted.Count) - 1))
    return [double]$sorted[$index]
}

function Get-ABSummary {
    param([object[]]$Pairs, [int]$Replicates = 20000)
    $rng = [Random]::new(1729)
    $rows = @()
    $names = @($Pairs[0].A.metrics.Keys)
    foreach ($name in $names) {
        $a = [double[]]@($Pairs | ForEach-Object { $_.A.metrics[$name].value })
        $b = [double[]]@($Pairs | ForEach-Object { $_.B.metrics[$name].value })
        if (@($a + $b | Where-Object { -not [double]::IsFinite($_) -or $_ -le 0 }).Count) {
            throw "Missing, zero or invalid metric: $name"
        }
        $logs = [double[]]@(for ($i = 0; $i -lt $a.Count; $i++) { [Math]::Log($b[$i] / $a[$i]) })
        $meanLog = ($logs | Measure-Object -Average).Average
        $sumSquares = 0.0
        foreach ($value in $logs) { $sumSquares += ($value - $meanLog) * ($value - $meanLog) }
        $standardError = [Math]::Sqrt($sumSquares / ($logs.Count - 1) / $logs.Count)
        $samples = [double[]]::new($Replicates)
        for ($j = 0; $j -lt $Replicates; $j++) {
            $draw = [double[]]@(for ($i = 0; $i -lt $a.Count; $i++) { $logs[$rng.Next($a.Count)] })
            $samples[$j] = [Math]::Exp((Get-ABMedian $draw))
        }
        $low = Get-ABQuantile $samples 0.025
        $high = Get-ABQuantile $samples 0.975
        $bootstrapCI = @($low, $high)
        # df=2 is conservative for every admitted n>=3 under the normal log-ratio model.
        $u = 0.95
        $critical = $u * [Math]::Sqrt(2 / (1 - $u * $u))
        $low = [Math]::Min($low, [Math]::Exp($meanLog - $critical * $standardError))
        $high = [Math]::Max($high, [Math]::Exp($meanLog + $critical * $standardError))
        $higher = $Pairs[0].A.metrics[$name].higher_is_better
        # Noninferiority requires excluding the regression margin, not a nonsignificant test.
        $pass = if ($higher) { $low -ge 0.97 } else { $high -le 1.03 }
        $winLow = Get-ABQuantile $samples (0.025 / ($names.Count * 4))
        $winHigh = Get-ABQuantile $samples (1 - 0.025 / ($names.Count * 4))
        $winU = 1 - 0.05 / ($names.Count * 4)
        $winCritical = $winU * [Math]::Sqrt(2 / (1 - $winU * $winU))
        $winLow = [Math]::Min($winLow, [Math]::Exp($meanLog - $winCritical * $standardError))
        $winHigh = [Math]::Max($winHigh, [Math]::Exp($meanLog + $winCritical * $standardError))
        $win = if ($higher) { $winLow -gt 1 } else { $winHigh -lt 1 }
        $regression = if ($higher) { $high -lt 0.97 } else { $low -gt 1.03 }
        $rows += [ordered]@{
            metric = $name; higher_is_better = $higher; pairs = $a.Count
            A_median = Get-ABMedian $a; B_median = Get-ABMedian $b
            A_min = ($a | Measure-Object -Minimum).Minimum; A_max = ($a | Measure-Object -Maximum).Maximum
            B_min = ($b | Measure-Object -Minimum).Minimum; B_max = ($b | Measure-Object -Maximum).Maximum
            A_iqr = (Get-ABQuantile $a 0.75) - (Get-ABQuantile $a 0.25)
            B_iqr = (Get-ABQuantile $b 0.75) - (Get-ABQuantile $b 0.25)
            B_over_A = [Math]::Exp((Get-ABMedian $logs))
            bootstrap_ci95 = $bootstrapCI
            ci95 = @($low, $high); noninferior = $pass; win = ($pass -and $win)
            win_family_ci = @($winLow, $winHigh)
            status = if ($pass) { if ($win) { 'win' } else { 'noninferior' } } elseif ($regression) { 'regression' } else { 'inconclusive' }
        }
    }
    return $rows
}

function New-ABPrompt {
    param([int]$Tokens, [string]$Salt, [string]$Purpose)
    $nonce = (Get-ABHash $Salt).Substring(0,32)
    $content = "/no_think nonce $nonce. " + (' hello' * [Math]::Max(0, $Tokens - 64)) + "`n$Purpose"
    return @(
        @{ role = 'system'; content = "Isolated benchmark nonce $nonce. Answer directly." },
        @{ role = 'user'; content = $content }
    )
}

function Add-ABMetric {
    param($Battery, [string]$Name, [double]$Value, [bool]$Higher = $true)
    if (-not [double]::IsFinite($Value) -or $Value -le 0) { throw "Invalid metric $Name=$Value" }
    $Battery.metrics[$Name] = @{ value = $Value; higher_is_better = $Higher }
}

function Invoke-ABNonStreamPair {
    param([object[]]$ChatMessages, [object[]]$AgentMessages, [int]$Limit = 128, [int]$MinimumTokens = 1)
    $requests = @()
    $started = [Diagnostics.Stopwatch]::StartNew()
    try {
        foreach ($messages in @(@{ messages = $ChatMessages }, @{ messages = $AgentMessages })) {
            $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$BaseUrl/chat/completions")
            $request.Content = [Net.Http.StringContent]::new((New-BodyJson $messages.messages $Limit $false),
                [Text.Encoding]::UTF8, 'application/json')
            $requests += @{ request = $request; task = $client.SendAsync($request); response = $null; wall = $null }
        }
        $pending = @(0,1)
        while ($pending.Count) {
            $index = [Threading.Tasks.Task]::WaitAny([Threading.Tasks.Task[]]@($pending | ForEach-Object { $requests[$_].task }))
            $which = $pending[$index]
            $requests[$which].response = $requests[$which].task.GetAwaiter().GetResult()
            $requests[$which].wall = $started.Elapsed.TotalSeconds
            $pending = @($pending | Where-Object { $_ -ne $which })
        }
        $rows = @()
        foreach ($item in $requests) {
            $null = $item.response.EnsureSuccessStatusCode()
            $payload = $item.response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
            $headers = Get-HeaderMap $item.response
            $decode = Get-HeaderNumber $headers 'X-Flash-Gordon-Decode-Seconds'
            $tokens = [int]$payload.usage.completion_tokens
            if ($tokens -lt $MinimumTokens -or -not $decode -or $decode -le 0) { throw 'Concurrent sample lacks sustained actual-token accounting.' }
            $rows += [ordered]@{
                completion_tokens = $tokens; wall_seconds = $item.wall; decode_seconds = $decode
                raw_end_to_end_tps = $tokens / $item.wall; engine_tps = $tokens / $decode
            }
        }
        $length = ($rows.completion_tokens | Measure-Object -Minimum).Minimum
        foreach ($row in $rows) {
            $row.normalized_wall_seconds = $row.wall_seconds - $row.decode_seconds + $row.decode_seconds * $length / $row.completion_tokens
            $row.normalized_end_to_end_tps = $length / $row.normalized_wall_seconds
        }
        return [ordered]@{
            streams = $rows; common_tokens = $length
            raw_aggregate_tps = ($rows.completion_tokens | Measure-Object -Sum).Sum / ($rows.wall_seconds | Measure-Object -Maximum).Maximum
            normalized_aggregate_tps = 2 * $length / ($rows.normalized_wall_seconds | Measure-Object -Maximum).Maximum
            normalization = 'Retain observed non-decode/queue time; scale each engine decode interval to minimum actual completion count. Approximation, not fixed-token execution.'
        }
    } finally {
        foreach ($item in $requests) {
            if ($item.response) { $item.response.Dispose() }
            $item.request.Dispose()
        }
    }
}

function Invoke-ABBattery {
    param([string]$Salt, [int[]]$Contexts = @(4096,32768,131072), [int]$DecodeTokens = 128)
    $battery = [ordered]@{ metrics = [ordered]@{}; prefill = @(); decode = @(); hashes = @(); requests = @() }
    $purpose = 'Write a long numbered list of distinct practical software tasks. Continue until the token limit.'
    foreach ($context in $Contexts) {
        $messages = New-ABPrompt $context "$Salt/context/$context" 'Output exactly Ready.'
        $bodyHash = Get-ABHash (New-BodyJson $messages 1 $false)
        $cold = Invoke-NonStream $messages 1
        $reused = Get-HeaderNumber $cold.Headers 'X-Flash-Gordon-Reused-Tokens'
        if ($null -eq $reused -or $reused -ne 0) { throw "Cold $context reused tokens: $reused" }
        $engine = Get-HeaderNumber $cold.Headers 'X-Flash-Gordon-Prefill-TPS'
        $battery.prefill += @{ target = $context; prompt_tokens = $cold.PromptTokens; request_sha256 = $bodyHash; headers = $cold.Headers; wall_seconds = $cold.WallSeconds }
        Add-ABMetric $battery "prefill-$context.engine_tps" $engine
        Add-ABMetric $battery "prefill-$context.client_wall_tps" ($cold.PromptTokens / $cold.WallSeconds)
        $follow = @($messages) + @(
            @{ role = 'assistant'; content = $cold.Content },
            @{ role = 'user'; content = "/no_think $purpose" }
        )
        $decode = Invoke-NonStream $follow $DecodeTokens
        $decodeReused = Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Reused-Tokens'
        if ($null -eq $decodeReused -or $decodeReused -lt ($cold.PromptTokens - 8)) {
            throw "Long decode did not retain its exact $context prefix."
        }
        $prefillSeconds = Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Prefill-Seconds'
        $decodeSeconds = $decode.WallSeconds - $prefillSeconds
        if ($decode.CompletionTokens -lt 32 -or $decodeSeconds -le 0) { throw "Insufficient sustained decode at $context." }
        $battery.decode += @{ context = $context; tokens = $decode.CompletionTokens; request_sha256 = Get-ABHash (New-BodyJson $follow $DecodeTokens $false); headers = $decode.Headers }
        Add-ABMetric $battery "decode-$context.engine_tps" (Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Decode-TPS')
        Add-ABMetric $battery "decode-$context.client_tps" ($decode.CompletionTokens / $decodeSeconds)
        Write-Host "AB context=$context prompt_tokens=$($cold.PromptTokens) cold_wall_s=$($cold.WallSeconds) decode_tokens=$($decode.CompletionTokens)"
    }
    $short = New-ABPrompt 64 "$Salt/short" $purpose
    $decode = Invoke-NonStream $short $DecodeTokens
    $prefillSeconds = Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Prefill-Seconds'
    if ($decode.CompletionTokens -lt 32) { throw 'Short decode stopped before 32 actual tokens.' }
    Add-ABMetric $battery 'decode-short.engine_tps' (Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Decode-TPS')
    Add-ABMetric $battery 'decode-short.client_tps' ($decode.CompletionTokens / ($decode.WallSeconds - $prefillSeconds))
    $chat = New-ABPrompt 0 "$Salt/chat" 'Say hello briefly.'
    $ttft = Invoke-Ttft 'chat' $chat 30
    Add-ABMetric $battery 'chat.ttft_ms' $ttft.ttft_ms $false
    $prefix = New-ABPrompt 3900 "$Salt/prefix" 'Output exactly Ready.'
    $prefixPrime = Invoke-NonStream $prefix 1
    $cached = @($prefix) + @(
        @{ role = 'assistant'; content = $prefixPrime.Content },
        @{ role = 'user'; content = '/no_think State the next action briefly.' }
    )
    $ttft = Invoke-Ttft 'cached' $cached 32
    $battery.requests += Get-ABHash (New-BodyJson $cached 32 $true)
    Add-ABMetric $battery 'cached-prefix.ttft_ms' $ttft.ttft_ms $false
    $concurrentChat = New-ABPrompt 64 "$Salt/concurrent-chat" $purpose
    $concurrentAgent = New-ABPrompt 64 "$Salt/concurrent-agent" $purpose
    $battery.concurrency = Invoke-ABNonStreamPair $concurrentChat $concurrentAgent $DecodeTokens -MinimumTokens 32
    Add-ABMetric $battery 'concurrent-chat.normalized_tps' $battery.concurrency.streams[0].normalized_end_to_end_tps
    Add-ABMetric $battery 'concurrent-agent.normalized_tps' $battery.concurrency.streams[1].normalized_end_to_end_tps
    Add-ABMetric $battery 'concurrent.normalized_aggregate_tps' $battery.concurrency.normalized_aggregate_tps
    $battery.hashes = @(Invoke-Correctness | ForEach-Object { @{ name = $_.name; sha256 = $_.sha256; tokens = $_.actual_tokens } })
    return $battery
}

function Invoke-PairedQualification {
    $config = Get-Content -Raw -LiteralPath $ABConfig | ConvertFrom-Json -AsHashtable
    foreach ($field in @('A','B','deploy_script','restore_script','change_kind')) {
        if (-not $config.ContainsKey($field)) { throw "A/B configuration missing $field" }
    }
    if ($WriteReference) { throw 'A/B never overwrites historical references.' }
    if ($config.change_kind -notin @('exact','weight-format')) { throw 'Unknown change_kind.' }
    if ($config.ContainsKey('required_exact_control')) {
        $control = Get-Content -Raw -LiteralPath $config.required_exact_control | ConvertFrom-Json -AsHashtable
        if (-not $control.complete -or -not $control.hash_identity) {
            throw 'The required exact-math binary control did not pass greedy identity.'
        }
    }
    if ($config.change_kind -eq 'weight-format') {
        $receipt = Get-Content -Raw -LiteralPath $config.quality_file | ConvertFrom-Json -AsHashtable
        $quality = Get-Content -Raw -LiteralPath $receipt.quality_path | ConvertFrom-Json -AsHashtable
        $qualityHash = (Get-FileHash -LiteralPath $receipt.quality_path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($receipt.manifest_sha256 -ne $config.B.manifest_sha256 -or $qualityHash -ne $receipt.quality_sha256 -or
            -not $quality.passed -or $quality.samples -lt 400 -or $quality.full_vocab -ne 248320 -or
            $quality.mean_kld -gt 0.01 -or $quality.p99_kld -gt 0.05 -or $quality.top1_agreement -lt 0.97) {
            throw 'Weight-format candidate lacks passing manifest-bound KL evidence.'
        }
    }
    $referenceRun = $null
    if ($config.ContainsKey('exact_hash_reference')) {
        $referenceRun = Get-Content -Raw -LiteralPath $config.exact_hash_reference | ConvertFrom-Json -AsHashtable
        if (-not $referenceRun.complete -or -not $referenceRun.quality_pass -or
            $referenceRun.config.B.manifest_sha256 -ne $config.B.manifest_sha256) {
            throw 'Exact hash reference must be a complete, quality-approved same-pack run.'
        }
    }
    $destination = if ($ABOutput) { $ABOutput } else { Join-Path $runRoot ("ab-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
    if ((Test-Path -LiteralPath $destination) -and @(Get-ChildItem -LiteralPath $destination -Force).Count) {
        throw 'A/B output directory must be fresh.'
    }
    $null = New-Item -ItemType Directory -Path $destination -Force
    $result = [ordered]@{
        schema = 1; started = [DateTimeOffset]::Now.ToString('o'); config = $config
        method = 'Paired round percentile bootstrap, median log B/A, 20000 fixed-seed resamples; expand CI with conservative df=2 Student-t log-ratio interval. Marginal 95% intersection-union noninferiority. Bonferroni win CI across metrics and four predeclared candidates. Guard assumes independent normal log ratios; bootstrap alone is exploratory at small n.'
        rounds = $ABRounds; order = $ABOrder; pairs = @(); metrics = @(); complete = $false; eligible = $false
        warning = 'Three paired rounds are low power. Inconclusive is not noninferior. Repeated tokens are not independent replicates.'
    }
    $path = Join-Path $destination 'ab-result.json'
    try {
        for ($round = 0; $round -lt $ABRounds; $round++) {
            $salt = [Guid]::NewGuid().ToString('N')
            $pair = [ordered]@{ round = $round; salt = $salt; A = $null; B = $null }
            $order = if ($ABOrder -eq 'ABBA' -and $round % 2) { @('B','A') } else { @('A','B') }
            foreach ($arm in $order) {
                Write-Host "AB round=$round arm=$arm deploying"
                & $config.deploy_script -Deployment $config[$arm] -Arm $arm
                if (-not $?) { throw "Deployment $arm failed." }
                $records.metadata.model = Get-Model
                $contexts = if ($config.ContainsKey('contexts')) { [int[]]$config.contexts } else { @(4096,32768,131072) }
                $pair[$arm] = Invoke-ABBattery $salt $contexts
                $pair[$arm] | ConvertTo-Json -Depth 45 | Set-Content (Join-Path $destination "round-$round-$arm.json")
                if ($config.ContainsKey('snapshot_script')) {
                    & $config.snapshot_script -Deployment $config[$arm] -Arm $arm -SnapshotOnly
                    if (-not $?) { throw 'Post-sample hardware snapshot failed.' }
                }
            }
            for ($i = 0; $i -lt $pair.A.prefill.Count; $i++) {
                if ($pair.A.prefill[$i].request_sha256 -ne $pair.B.prefill[$i].request_sha256) { throw 'Paired request bytes differ.' }
                if ($pair.A.decode[$i].request_sha256 -ne $pair.B.decode[$i].request_sha256) { throw 'Paired decode request bytes differ.' }
            }
            if (($pair.A.requests -join ',') -ne ($pair.B.requests -join ',')) { throw 'Paired cached request bytes differ.' }
            $pair.hash_match = @(
                for ($i = 0; $i -lt $pair.A.hashes.Count; $i++) {
                    @{ name = $pair.A.hashes[$i].name; match = $pair.A.hashes[$i].sha256 -eq $pair.B.hashes[$i].sha256 }
                }
            )
            $result.pairs += $pair
            $result | ConvertTo-Json -Depth 55 | Set-Content $path
        }
        $result.metrics = @(Get-ABSummary $result.pairs)
        $hashPass = @($result.pairs | ForEach-Object { $_.hash_match } | Where-Object { -not $_.match }).Count -eq 0
        $result.hash_identity = $hashPass
        $result.quality_pass = $config.change_kind -eq 'weight-format' -or $hashPass
        if ($config.ContainsKey('exact_hash_reference')) {
            $referenceHashes = @{}
            foreach ($hash in $referenceRun.pairs[0].B.hashes) { $referenceHashes[$hash.name] = $hash.sha256 }
            $samePackPass = @($result.pairs | ForEach-Object { $_.B.hashes } |
                Where-Object { $referenceHashes[$_.name] -ne $_.sha256 }).Count -eq 0
            $result.same_pack_exact_hash_identity = $samePackPass
            $result.quality_pass = $result.quality_pass -and $samePackPass
        }
        $result.complete = $true
        $result.eligible = $result.quality_pass -and @($result.metrics | Where-Object { -not $_.noninferior }).Count -eq 0 -and @($result.metrics | Where-Object win).Count -gt 0
        Write-Host "AB result=$path eligible=$($result.eligible)"
    } catch {
        $result.error = $_.Exception.Message
        throw
    } finally {
        try {
            & $config.restore_script
            if (-not $?) { throw 'Original deployment restore failed.' }
            $result.restored_original = $true
        } catch {
            $result.restore_error = $_.Exception.Message
            $result.eligible = $false
            throw
        } finally {
            $result.finished = [DateTimeOffset]::Now.ToString('o')
            $result | ConvertTo-Json -Depth 55 | Set-Content $path
        }
    }
}
