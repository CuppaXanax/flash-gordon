. "$PSScriptRoot\ab-stack-common.ps1"

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

function Get-ABPrimaryMetrics {
    @('decode-short.engine_tps','decode-32768.engine_tps',
      'prefill-4096.engine_tps','prefill-32768.engine_tps',
      'chat.ttft_ms','concurrent.normalized_aggregate_tps')
}

function Get-ABSummary {
    param([object[]]$Pairs, [int]$Replicates = 20000,
          [string[]]$PrimaryMetrics = (Get-ABPrimaryMetrics))
    if ($Pairs.Count -lt 5) { throw 'Policy v2 requires at least five complete pairs.' }
    if ($Replicates -lt 100) { throw 'Insufficient bootstrap replicates.' }
    $rows = @()
    $names = @($Pairs[0].A.metrics.Keys | Sort-Object)
    foreach ($name in $names) {
        $rng = [Random]::new(1729)
        $higher = [bool]$Pairs[0].A.metrics[$name].higher_is_better
        foreach ($pair in $Pairs) {
            foreach ($arm in @('A','B')) {
                if (-not $pair[$arm].metrics.Contains($name) -or
                    [bool]$pair[$arm].metrics[$name].higher_is_better -ne $higher -or
                    (@($pair[$arm].metrics.Keys | Sort-Object) -join ',') -ne ($names -join ',')) {
                    throw "Paired metric set/direction changed: $name"
                }
            }
        }
        $a = [double[]]@($Pairs | ForEach-Object { $_.A.metrics[$name].value })
        $b = [double[]]@($Pairs | ForEach-Object { $_.B.metrics[$name].value })
        if (@($a + $b | Where-Object { -not [double]::IsFinite($_) -or $_ -le 0 }).Count) {
            throw "Missing, zero or invalid metric: $name"
        }
        $ratios = [double[]]@(for ($i = 0; $i -lt $a.Count; $i++) {
            if ($higher) { $b[$i] / $a[$i] } else { $a[$i] / $b[$i] }
        })
        $samples = [double[]]::new($Replicates)
        for ($j = 0; $j -lt $Replicates; $j++) {
            $draw = [double[]]@(for ($i = 0; $i -lt $a.Count; $i++) { $ratios[$rng.Next($a.Count)] })
            $samples[$j] = Get-ABMedian $draw
        }
        $low = Get-ABQuantile $samples 0.025
        $high = Get-ABQuantile $samples 0.975
        $median = Get-ABMedian $ratios
        $regression = $high -lt 0.97
        $pass = $median -ge 0.97 -and -not $regression
        $win = $pass -and $median -ge 1.03
        $rows += [ordered]@{
            metric = $name; higher_is_better = $higher; pairs = $a.Count
            A_median = Get-ABMedian $a; B_median = Get-ABMedian $b
            A_min = ($a | Measure-Object -Minimum).Minimum; A_max = ($a | Measure-Object -Maximum).Maximum
            B_min = ($b | Measure-Object -Minimum).Minimum; B_max = ($b | Measure-Object -Maximum).Maximum
            A_iqr = (Get-ABQuantile $a 0.75) - (Get-ABQuantile $a 0.25)
            B_iqr = (Get-ABQuantile $b 0.75) - (Get-ABQuantile $b 0.25)
            B_over_A = Get-ABMedian ([double[]]@(for ($i=0;$i-lt$a.Count;$i++){$b[$i]/$a[$i]}))
            benefit_ratio = $median; ratio_direction = if ($higher) { 'B/A' } else { 'A/B' }
            primary = $name -in $PrimaryMetrics
            bootstrap_ci95 = @($low, $high); ci95 = @($low, $high)
            passes_primary_guard = $pass; noninferior = $low -ge 0.97
            win = $win; significant_regression = $regression
            status = if ($pass) {
                if ($win) { 'median-win' } elseif ($low -ge 0.97) { 'ci-noninferior' } else { 'accepted-uncertain' }
            } elseif ($regression) { 'regression' } else { 'median-below-margin' }
        }
    }
    return $rows
}

function Get-ABDecision {
    param([object[]]$Metrics, [bool]$QualityPassed,
          [string[]]$PrimaryMetrics = (Get-ABPrimaryMetrics))
    $present = @($Metrics | ForEach-Object { $_.metric })
    $missing = @($PrimaryMetrics | Where-Object { $_ -notin $present })
    $primary = @($Metrics | Where-Object { $_.metric -in $PrimaryMetrics })
    $pass = -not $missing.Count -and $primary.Count -eq $PrimaryMetrics.Count -and
        @($primary | Where-Object { $_.pairs -lt 5 -or -not $_.passes_primary_guard }).Count -eq 0
    $win = @($primary | Where-Object win).Count -gt 0
    return @{ primary_metrics = $PrimaryMetrics; missing_primary_metrics = $missing
        all_primary_metrics_pass = $pass; primary_improvement = $win
        eligible = $QualityPassed -and $pass -and $win }
}

function Test-ABCanonicalHashes {
    param([object[]]$Reference, [object[]]$Actual)
    if (-not $Reference.Count -or $Reference.Count -ne $Actual.Count) { return $false }
    $expected = @{}; $seen = @{}
    foreach ($row in $Reference) {
        if ($expected.ContainsKey($row.name)) { return $false }
        $expected[$row.name] = $row.sha256
    }
    foreach ($row in $Actual) {
        if ($seen.ContainsKey($row.name) -or -not $expected.ContainsKey($row.name) -or
            $expected[$row.name] -ne $row.sha256) { return $false }
        $seen[$row.name] = $true
    }
    return $true
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
                name = if ($rows.Count -eq 0) { 'chat-style' } else { 'agent-style' }
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
            schema = 2; token_count_source = 'actual usage on these exact non-stream responses'
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
    param([string]$Salt, [int[]]$Contexts = @(4096,32768), [int]$DecodeTokens = 128, [switch]$LongOnly)
    $battery = [ordered]@{ metrics = [ordered]@{}; prefill = @(); decode = @(); hashes = @(); requests = @() }
    $purpose = 'Write a long numbered list of distinct practical software tasks. Continue until the token limit.'
    foreach ($context in $Contexts) {
        $messages = New-ABPrompt $context "$Salt/context/$context" 'Output exactly Ready.'
        $bodyHash = Get-ABHash (New-BodyJson $messages 1 $false)
        $cold = Invoke-NonStream $messages 1
        $reused = Get-HeaderNumber $cold.Headers 'X-Flash-Gordon-Reused-Tokens'
        if ($null -eq $reused -or $reused -ne 0) { throw "Cold $context reused tokens: $reused" }
        $engine = Get-HeaderNumber $cold.Headers 'X-Flash-Gordon-Prefill-TPS'
        $battery.prefill += @{ target = $context; prompt_tokens = $cold.PromptTokens; request_sha256 = $bodyHash; headers = $cold.Headers; wall_seconds = $cold.WallSeconds; generated_text = $cold.Content }
        Add-ABMetric $battery "prefill-$context.engine_tps" $engine
        Add-ABMetric $battery "prefill-$context.client_wall_tps" ($cold.PromptTokens / $cold.WallSeconds)
        $follow = @($messages) + @(
            @{ role = 'assistant'; content = 'Ready' },
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
    if ($LongOnly) {
        $battery.hashes = @(Invoke-Correctness | ForEach-Object { @{ name = $_.name; sha256 = $_.sha256; tokens = $_.actual_tokens } })
        return $battery
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
        @{ role = 'assistant'; content = 'Ready' },
        @{ role = 'user'; content = '/no_think State the next action briefly.' }
    )
    $ttft = Invoke-Ttft 'cached' $cached 32
    $battery.requests += Get-ABHash (New-BodyJson $cached 32 $true)
    $battery.prefix_prime_text = $prefixPrime.Content
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
    if ($config.change_kind -notin @('exact','weight-format','numeric')) { throw 'Unknown change_kind.' }
    if ($config.ContainsKey('required_exact_control')) {
        $control = Get-Content -Raw -LiteralPath $config.required_exact_control | ConvertFrom-Json -AsHashtable
        if (-not $control.complete -or -not $control.hash_identity) {
            throw 'The required exact-math binary control did not pass greedy identity.'
        }
    }
    $qualityEvidenceHash = ''
    if ($config.change_kind -in @('weight-format','numeric')) {
        $qualityEvidenceHash = (Get-FileHash -LiteralPath $config.quality_file -Algorithm SHA256).Hash.ToLowerInvariant()
        $receipt = Get-Content -Raw -LiteralPath $config.quality_file | ConvertFrom-Json -AsHashtable
        $quality = Get-Content -Raw -LiteralPath $receipt.quality_path | ConvertFrom-Json -AsHashtable
        $qualityHash = (Get-FileHash -LiteralPath $receipt.quality_path -Algorithm SHA256).Hash.ToLowerInvariant()
        foreach ($field in @('mean_kld','p99_kld','top1_agreement')) {
            if (-not $quality.ContainsKey($field) -or -not [double]::IsFinite([double]$quality[$field]) -or
                [double]$quality[$field] -lt 0) { throw "Invalid KL evidence field: $field" }
        }
        if ($receipt.manifest_sha256 -ne $config.B.manifest_sha256 -or $qualityHash -ne $receipt.quality_sha256 -or
            $quality.passed -isnot [bool] -or -not $quality.passed -or
            $quality.corpus_sha256 -notmatch '^[a-f0-9]{64}$' -or
            $quality.samples -lt 400 -or $quality.full_vocab -ne 248320 -or $quality.top1_agreement -gt 1 -or
            $quality.mean_kld -gt 0.01 -or $quality.p99_kld -gt 0.05 -or $quality.top1_agreement -lt 0.97) {
            throw 'Weight-format candidate lacks passing manifest-bound KL evidence.'
        }
        if ($config.change_kind -eq 'numeric' -and
            ($config.A['shader_tree_sha256'] -notmatch '^[a-f0-9]{64}$' -or
             $config.B['shader_tree_sha256'] -notmatch '^[a-f0-9]{64}$' -or
             $receipt.A_identity -ne (Get-ABDeploymentIdentity $config.A) -or
             $receipt.B_identity -ne (Get-ABDeploymentIdentity $config.B) -or
             $receipt.corpus_sha256 -ne $quality.corpus_sha256)) {
            throw 'Numeric stack quality receipt does not bind both binaries, packs, flags and corpus.'
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
    $candidateFlags = if ($config.B.ContainsKey('flags')) { Get-ABStackFlags $config.B } else { @{} }
    if ($candidateFlags['FG_DECODE_SUBMIT_GROUP'] -in @('1','2')) {
        if (-not $referenceRun -or @($referenceRun.pairs).Count -lt 5 -or
            $referenceRun.policy -ne 'primary-median-v2' -or
            (Get-ABKernelIdentity $referenceRun.config.B) -ne (Get-ABKernelIdentity $config.B)) {
            throw 'Submission grouping requires a completed identical-binary/shader/pack/other-flags greedy control.'
        }
        foreach ($pair in $referenceRun.pairs) {
            if (-not (Test-ABCanonicalHashes $referenceRun.pairs[0].B.hashes $pair.B.hashes)) {
                throw 'Submission-group greedy control is not deterministic across its paired rounds.'
            }
        }
    }
    $destination = if ($ABOutput) { $ABOutput } else { Join-Path $runRoot ("ab-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
    if (-not $ABResume -and (Test-Path -LiteralPath $destination) -and @(Get-ChildItem -LiteralPath $destination -Force).Count) {
        throw 'A/B output directory must be fresh.'
    }
    $null = New-Item -ItemType Directory -Path $destination -Force
    $method = 'Policy v2: >=5 pairs (default 6); 20000 fixed-seed paired-round percentile bootstrap draws of median benefit ratio (TPS B/A, latency A/B), marginal 95% CI; no t guard or multiplicity adjustment. Each primary median >=.97 and CI not wholly below .97; at least one primary median >=1.03; quality required. Operational risk policy, not proof of statistical noninferiority.'
    $result = [ordered]@{
        schema = 2; policy = 'primary-median-v2'; started = [DateTimeOffset]::Now.ToString('o'); config = $config
        config_sha256 = Get-ABHash ($config | ConvertTo-Json -Depth 35 -Compress)
        quality_receipt_sha256 = $qualityEvidenceHash
        method = $method
        rounds = $ABRounds; order = $ABOrder; pairs = @(); metrics = @(); complete = $false; eligible = $false
        warning = 'A CI crossing .97 may pass the operational median guard; uncertainty remains visible. This is not a formal noninferiority claim. Secondary metrics never select/veto; no repeat-until-pass.'
    }
    $path = Join-Path $destination 'ab-result.json'
    $firstRound = 0
    if ($ABResume) {
        $previous = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -AsHashtable
        if ($previous.schema -ne 2 -or $previous.config_sha256 -ne $result.config_sha256 -or
            $previous.quality_receipt_sha256 -ne $qualityEvidenceHash -or
            $previous.complete -or $previous.rounds -ne $ABRounds -or $previous.order -ne $ABOrder -or
            $previous.config.change_kind -ne $config.change_kind) { throw 'Cannot change or repeat a completed A/B design.' }
        foreach ($arm in @('A','B')) {
            foreach ($field in @('binary_sha256','manifest_sha256','release','manifest','sessions','wide')) {
                if ($previous.config[$arm][$field] -ne $config[$arm][$field]) { throw "Resume changed $arm.$field" }
            }
        }
        if (($previous.config['contexts'] -join ',') -ne ($config['contexts'] -join ',')) { throw 'Resume changed contexts.' }
        foreach ($field in @('long_only','requires_long_qualification')) {
            if ([bool]$previous.config[$field] -ne [bool]$config[$field]) { throw "Resume changed $field" }
        }
        $firstRound = @($previous.pairs).Count
        if ($firstRound -ge $ABRounds) { throw 'All round samples already exist; analyze them without resampling.' }
        for ($i = 0; $i -lt $firstRound; $i++) {
            if ($previous.pairs[$i].round -ne $i -or -not $previous.pairs[$i].A -or -not $previous.pairs[$i].B) {
                throw 'Incomplete committed pair in resume record.'
            }
        }
        $previous.resume_attempts = @($previous['resume_attempts'] | Where-Object { $null -ne $_ }) + @(@{
            timestamp = [DateTimeOffset]::Now.ToString('o'); first_round = $firstRound
            prior_error = $previous['error']; prior_restore_error = $previous['restore_error']
        })
        $previous.Remove('error'); $previous.Remove('restore_error'); $previous.Remove('finished')
        $previous.method = $method; $previous.config = $config; $previous.restored_baseline = $false
        $result = $previous
        foreach ($partial in @(Get-ChildItem -LiteralPath $destination -Filter "round-$firstRound-*.json")) {
            Move-Item -LiteralPath $partial.FullName -Destination ($partial.FullName + '.interrupted-' + (Get-Date -Format 'HHmmss'))
        }
    }
    try {
        for ($round = $firstRound; $round -lt $ABRounds; $round++) {
            $salt = [Guid]::NewGuid().ToString('N')
            $pair = [ordered]@{ round = $round; salt = $salt; A = $null; B = $null }
            $order = if ($ABOrder -eq 'ABBA' -and $round % 2) { @('B','A') } else { @('A','B') }
            foreach ($arm in $order) {
                Write-Host "AB round=$round arm=$arm deploying"
                & $config.deploy_script -Deployment $config[$arm] -Arm $arm
                if (-not $?) { throw "Deployment $arm failed." }
                $records.metadata.model = Get-Model
                $contexts = if ($config.ContainsKey('contexts')) { [int[]]$config.contexts } else { @(4096,32768) }
                $pair[$arm] = Invoke-ABBattery $salt $contexts -LongOnly:($config.ContainsKey('long_only') -and $config.long_only)
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
        $result.quality_pass = $config.change_kind -in @('weight-format','numeric') -or $hashPass
        if ($config.ContainsKey('exact_hash_reference')) {
            $samePackPass = @($result.pairs | Where-Object {
                -not (Test-ABCanonicalHashes $referenceRun.pairs[0].B.hashes $_.B.hashes)
            }).Count -eq 0
            $result.same_pack_exact_hash_identity = $samePackPass
            if ($candidateFlags['FG_DECODE_SUBMIT_GROUP'] -in @('1','2')) {
                $result.same_kernel_hash_identity = $samePackPass
            }
            $result.quality_pass = $result.quality_pass -and $samePackPass
        }
        $result.complete = $true
        $decision = Get-ABDecision $result.metrics $result.quality_pass
        foreach ($key in $decision.Keys) { $result[$key] = $decision[$key] }
        if (($config.ContainsKey('requires_long_qualification') -and $config.requires_long_qualification) -or
            ($config.ContainsKey('long_only') -and $config.long_only)) {
            $result.partial_qualification = $true
            $result.eligible = $false
        }
        Write-Host "AB result=$path eligible=$($result.eligible)"
    } catch {
        $result.error = $_.Exception.Message
        throw
    } finally {
        try {
            & $config.restore_script
            if (-not $?) { throw 'Baseline deployment restore failed.' }
            $result.restored_baseline = $true
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
