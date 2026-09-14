param(
    [string]$BaseUrl = "http://192.0.2.42:8080",
    [int[]]$Contexts = @(4096, 16384, 32768, 65536, 131072, 262144),
    [int]$DecodeTokens = 32,
    [int]$TimeoutSeconds = 7200,
    [string]$ResultRoot = "results"
)

$ErrorActionPreference = "Continue"
$invariant = [Globalization.CultureInfo]::InvariantCulture
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $ResultRoot "context-sweep-$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$csv = Join-Path $outDir "sweep.csv"
Write-Host "context sweep -> $outDir"

function Get-HeaderValue {
    param($Response, [string]$Name)
    $value = [string]$Response.Headers[$Name]
    if (-not $value) { throw "Response omitted header $Name" }
    return $value
}

function Send-Prompt {
    param([string]$Content, [int]$MaxTokens, [int]$Seed)
    $body = @{
        model = "Qwen3.8-Flash-Next"
        messages = @(@{ role = "user"; content = $Content })
        temperature = 0.0
        top_p = 1.0
        top_k = 1
        seed = $Seed
        max_tokens = $MaxTokens
        stream = $false
    } | ConvertTo-Json -Depth 12 -Compress
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec $TimeoutSeconds
    $watch.Stop()
    [pscustomobject]@{
        WallSeconds      = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
        PromptTokens     = [int](Get-HeaderValue $r "X-Flash-Gordon-Prompt-Tokens")
        CompletionTokens = [int](Get-HeaderValue $r "X-Flash-Gordon-Completion-Tokens")
        PrefillSeconds   = [double]::Parse((Get-HeaderValue $r "X-Flash-Gordon-Prefill-Seconds"), $invariant)
        PrefillTps       = [double]::Parse((Get-HeaderValue $r "X-Flash-Gordon-Prefill-TPS"), $invariant)
        DecodeTps        = [double]::Parse((Get-HeaderValue $r "X-Flash-Gordon-Decode-TPS"), $invariant)
        Mode             = [string](Get-HeaderValue $r "X-Flash-Gordon-Execution-Mode")
    }
}

function Add-Row {
    param([int]$Target, [string]$Phase, $Result, [string]$Error)
    if ($Result) {
        $row = [pscustomobject]@{
            Target = $Target; Phase = $Phase; WallSeconds = $Result.WallSeconds
            PromptTokens = $Result.PromptTokens; CompletionTokens = $Result.CompletionTokens
            PrefillSeconds = $Result.PrefillSeconds; PrefillTps = $Result.PrefillTps
            DecodeTps = $Result.DecodeTps; Mode = $Result.Mode; Error = ""
        }
    } else {
        $row = [pscustomobject]@{
            Target = $Target; Phase = $Phase; WallSeconds = 0
            PromptTokens = 0; CompletionTokens = 0
            PrefillSeconds = 0; PrefillTps = 0
            DecodeTps = 0; Mode = ""; Error = $Error
        }
    }
    return $row
}

$rows = [Collections.Generic.List[object]]::new()
foreach ($ctx in ($Contexts | Sort-Object)) {
    if ($ctx -lt 64) { continue }
    $repeats = [Math]::Max(32, $ctx - 48)
    $prompt = "/no_think " + ("hello " * $repeats) + "Reply with one word."
    Write-Host ("--- ctx target {0} (repeats {1}) ---" -f $ctx, $repeats)

    try {
        $p = Send-Prompt -Content $prompt -MaxTokens 1 -Seed 7400
        $row = Add-Row -Target $ctx -Phase "prefill" -Result $p -Error ""
        Write-Host ("ctx={0,7} prefill: prompt={1,7} {2,8:F2} TPS ({3,8:F2}s) wall={4,8:F2}s" -f `
            $ctx, $p.PromptTokens, $p.PrefillTps, $p.PrefillSeconds, $p.WallSeconds)
    } catch {
        $row = Add-Row -Target $ctx -Phase "prefill" -Result $null -Error $_.Exception.Message
        Write-Host "ctx=$ctx prefill FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
    $rows.Add($row); $rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding utf8

    try {
        $d = Send-Prompt -Content $prompt -MaxTokens $DecodeTokens -Seed 7401
        $row = Add-Row -Target $ctx -Phase "decode" -Result $d -Error ""
        Write-Host ("ctx={0,7} decode : prompt={1,7} prefill={2,8:F2} TPS decode={3,7:F2} TPS wall={4,8:F2}s" -f `
            $ctx, $d.PromptTokens, $d.PrefillTps, $d.DecodeTps, $d.WallSeconds)
    } catch {
        $row = Add-Row -Target $ctx -Phase "decode" -Result $null -Error $_.Exception.Message
        Write-Host "ctx=$ctx decode FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
    $rows.Add($row); $rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding utf8
}

$rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir "sweep.json") -Encoding utf8
Write-Host "Sweep complete: $outDir"
