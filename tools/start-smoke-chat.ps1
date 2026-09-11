#Requires -Version 7.4
<#
.SYNOPSIS
Build the current checkout, start eight EP blades, smoke-test, and chat.
.EXAMPLE
pwsh -File .\tools\start-smoke-chat.ps1
.EXAMPLE
pwsh -File .\tools\start-smoke-chat.ps1 -SmokeOnly -NoChat
.DESCRIPTION
Uses Invoke-BC250Fleet.ps1 and its .env/password authentication by default.
Requires existing EP model artifacts on .42-.49 and
make/cc/glslangValidator/Vulkan development files on blade 42. Builds the local
working files, including uncommitted changes, using ordinary make. Distributes
identical binary/shaders. Stops only your Flash Gordon processes using manifests
inside PackDirectory. Leaves the API running when you exit. Does not install
packages, repack weights, unlock CUs, or use the other four blades.
SmokeOnly checks an already running API, without SSH or deployment.
#>
[CmdletBinding()]
param(
    [string]$SshUser,
    [ValidateSet('Password','Key')][string]$Auth = 'Password',
    [string]$FleetScript = (Join-Path $PSScriptRoot '../../bc-250-dbg/Invoke-BC250Fleet.ps1'),
    [string]$IdentityFile = "$HOME/.ssh/bc250-admin",
    [string]$PackDirectory = '/home/user/flash-gordon-q38-cooked',
    [string]$ManifestName = 'manifest-native-262k.fgm',
    [string]$RemoteDirectory = '/home/user/flash-gordon-live',
    [string]$BaseUrl,
    [ValidateRange(1024,65535)][int]$Port = 8080,
    [ValidateRange(60,7200)][int]$StartupTimeoutSeconds = 1800,
    [ValidateRange(60,86400)][int]$RequestTimeoutSeconds = 1800,
    [switch]$SmokeOnly,
    [switch]$NoChat
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6)
$results = Join-Path $repo "results/smoke-$runId"
$null = New-Item -ItemType Directory -Path $results
$base = if ($BaseUrl) { $BaseUrl.TrimEnd('/') } else { "http://192.0.2.42:$Port/v1" }
$model = 'Qwen3.8-Flash-Next'
$targets = @(42..49 | ForEach-Object { "192.0.2.$_" })
$fleetArguments = @{ Auth=$Auth; IdentityFile=$IdentityFile; PassThru=$true }
if ($SshUser) { $fleetArguments.User = $SshUser }
$script:remoteSequence = 0

function Invoke-FleetStep([string]$Target, [string]$Script, [hashtable]$Transfer = @{}) {
    $script:remoteSequence++
    $scriptPath = Join-Path $results ("step-{0:D3}.sh" -f $script:remoteSequence)
    [IO.File]::WriteAllText($scriptPath, $Script.Replace("`r",''))
    Write-Host "[$Target] Fleet step $script:remoteSequence..."
    $records = @(& $FleetScript -ScriptPath $scriptPath -Targets @($Target) @fleetArguments @Transfer)
    if ($records.Count -ne 1 -or $records[0].IP -ne $Target) { throw "Fleet runner returned an incomplete result for $Target." }
    $record = $records[0]
    foreach ($text in @($record.Stdout, $record.Stderr)) {
        if ($text) {
            $text | Add-Content -LiteralPath (Join-Path $results "$Target.log")
            $text -split "\r?\n" | ForEach-Object { Write-Host "[$Target] $_" }
        }
    }
    if ($record.ExitCode -ne 0) { throw "Fleet step failed on $Target (exit $($record.ExitCode)). See $results." }
}
function Remote([string]$Target, [string]$Script) {
    Invoke-FleetStep $Target $Script
}
function Copy-Remote([string]$Source, [string]$Destination) {
    if ($Destination -match '^[^@]*@(?<target>[0-9.]+):(?<path>/.*)$') {
        Invoke-FleetStep $Matches.target 'true' @{UploadPath=$Source; UploadDestination=$Matches.path}
    } elseif ($Source -match '^[^@]*@(?<target>[0-9.]+):(?<path>/.*)$') {
        Invoke-FleetStep $Matches.target 'true' @{DownloadSource=$Matches.path; DownloadPath=$Destination}
    } else { throw 'Expected one local path and one fleet address for transfer.' }
}

if (!$SmokeOnly) {
    foreach ($command in @('tar','git')) { $null = Get-Command $command -ErrorAction Stop }
    if (!(Test-Path -LiteralPath $FleetScript)) { throw "Fleet runner not found: $FleetScript" }
    $fleetCommand = Get-Command $FleetScript
    foreach ($parameter in @('UploadPath','UploadDestination','DownloadSource','DownloadPath')) {
        if (!$fleetCommand.Parameters.ContainsKey($parameter)) { throw "Fleet runner needs the file-transfer update: $FleetScript" }
    }
    foreach ($path in @($PackDirectory,$RemoteDirectory)) {
        if ($path -notmatch '^/[A-Za-z0-9_./-]+$' -or $path -match '(^|/)\.\.(/|$)' -or $path -eq '/') { throw "Use a specific absolute Linux directory: $path" }
    }
    if ($ManifestName -notmatch '^[A-Za-z0-9_.-]+$' -or ($SshUser -and $SshUser -notmatch '^[A-Za-z0-9_-]+$')) { throw 'Invalid manifest name or SSH username.' }
    $branch = & git -C $repo branch --show-current
    Write-Host "Deploying current working files from $repo ($branch) to ranks 0-7."
    $release = "$RemoteDirectory/$runId"
    $liveManifest = "$PackDirectory/manifest-live-$runId.fgm"
    foreach ($target in $targets) {
        Remote $target "set -eu; test -r '$PackDirectory/$ManifestName'; mkdir -p '$release'"
    }
    $snapshot = Join-Path $results 'source'
    $null = New-Item -ItemType Directory -Path $snapshot
    $files = @('Makefile','glslc')
    foreach ($directory in @('src','include','shaders')) {
        $files += Get-ChildItem -LiteralPath (Join-Path $repo $directory) -File -Recurse |
            Where-Object Extension -In @('.c','.h','.comp') |
            ForEach-Object { [IO.Path]::GetRelativePath($repo, $_.FullName) }
    }
    foreach ($file in $files) {
        $destination = Join-Path $snapshot $file
        $null = New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force
        [IO.File]::WriteAllText($destination, [IO.File]::ReadAllText((Join-Path $repo $file)).Replace("`r`n","`n"))
    }
    $archive = Join-Path $results 'source.tar.gz'
    & tar -czf $archive -C $snapshot Makefile glslc src include shaders
    if ($LASTEXITCODE) { throw 'Source archive failed.' }
    Copy-Remote $archive "${SshUser}@$($targets[0]):$release/source.tar.gz"
    $build = @'
set -euo pipefail
cd '__RELEASE__'
tar -xzf source.tar.gz
chmod +x glslc
make -j"$(nproc)"
./flash-gordon upgrade-manifest --input '__PACK__/__MANIFEST__' --output '__LIVE__' --profile native-262k-microbatch-128
./flash-gordon inspect --manifest '__LIVE__'
tar -czf runtime.tar.gz flash-gordon vulkan
sha256sum flash-gordon runtime.tar.gz '__LIVE__'
'@
    Remote $targets[0] ($build.Replace('__RELEASE__',$release).Replace('__PACK__',$PackDirectory).Replace('__MANIFEST__',$ManifestName).Replace('__LIVE__',$liveManifest))
    $runtimeArchive = Join-Path $results 'runtime.tar.gz'
    $manifestCopy = Join-Path $results 'manifest.fgm'
    Copy-Remote "${SshUser}@$($targets[0]):$release/runtime.tar.gz" $runtimeArchive
    Copy-Remote "${SshUser}@$($targets[0]):$liveManifest" $manifestCopy
    $expectedArchive = (Get-FileHash -LiteralPath $runtimeArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedManifest = (Get-FileHash -LiteralPath $manifestCopy -Algorithm SHA256).Hash.ToLowerInvariant()
    foreach ($target in $targets | Select-Object -Skip 1) {
        Copy-Remote $runtimeArchive "${SshUser}@${target}:$release/runtime.tar.gz"
        Copy-Remote $manifestCopy "${SshUser}@${target}:$liveManifest"
        Remote $target "set -eu; cd '$release'; tar -xzf runtime.tar.gz"
    }
    foreach ($target in $targets) {
        Remote $target "set -eu; echo '$expectedArchive  $release/runtime.tar.gz' | sha256sum -c -; echo '$expectedManifest  $liveManifest' | sha256sum -c -; '$release/flash-gordon' inspect --manifest '$liveManifest' >/dev/null"
    }
    $stop = @'
set -euo pipefail
for proc in /proc/[0-9]*; do
    test -O "$proc" || continue
    exe=$(readlink "$proc/exe" 2>/dev/null) || continue
    case "${exe##*/}" in flash-gordon|flash-gordon-*) ;; *) continue ;; esac
    mapfile -d '' -t args < "$proc/cmdline" || continue
    for ((i=0; i+1<${#args[@]}; i++)); do
        if [[ ${args[i]} == --manifest && ${args[i+1]} == '__PACK__/'* ]]; then
            pid=${proc##*/}
            kill -TERM "$pid"
            for ((waited=0; waited<30; waited++)); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then echo "Process $pid did not stop; refusing forced kill" >&2; exit 1; fi
            break
        fi
    done
done
'@
    foreach ($target in $targets) { Remote $target ($stop.Replace('__PACK__',$PackDirectory)) }
    foreach ($rank in @(1..7) + @(0)) {
        $arguments = if ($rank -eq 0) { "api --manifest '$liveManifest' --host 0.0.0.0 --port $Port" } else { "rank --manifest '$liveManifest' --rank $rank" }
        Remote $targets[$rank] "set -eu; cd '$release'; nohup ./flash-gordon $arguments >server.log 2>&1 < /dev/null & echo `$! >server.pid; echo 'Started rank $rank; log: $release/server.log'"
    }
    Write-Host "Waiting for model loading. Logs on each blade: $release/server.log"
    $until = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    $ready = $false
    do {
        try { $null = Invoke-RestMethod "$base/models" -TimeoutSec 5; $ready = $true } catch { Start-Sleep -Seconds 3 }
    } until ($ready -or [DateTime]::UtcNow -ge $until)
    if (!$ready) { throw "API startup timed out. Processes remain in place; inspect $release/server.log." }
}

$script:requestNumber = 0
function Request-Turn([object[]]$Messages, [int]$Limit) {
    $body = @{ model=$model; messages=$Messages; temperature=0; max_tokens=$Limit; stream=$false }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-WebRequest "$base/chat/completions" -Method Post -ContentType 'application/json' `
        -Body ([Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json -Depth 20 -Compress))) -TimeoutSec $RequestTimeoutSeconds
    $watch.Stop()
    $responseText = if ($response.Content -is [byte[]]) { [Text.Encoding]::UTF8.GetString($response.Content) } else { [string]$response.Content }
    $script:requestNumber++
    $responseText | Set-Content -LiteralPath (Join-Path $results "request-$script:requestNumber.json") -Encoding utf8
    $response.Headers | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $results "request-$script:requestNumber-headers.json") -Encoding utf8
    $completion = $responseText | ConvertFrom-Json -AsHashtable
    $answer = $completion.choices[0].message
    if ([string]::IsNullOrWhiteSpace($answer.content)) { throw 'The server returned no assistant text; it may have exhausted the token limit while thinking. The raw response and timing headers were saved in the results directory.' }
    if ([string]$response.Headers['X-Flash-Gordon-Execution-Mode'] -ne 'expert-parallel') { throw 'The response did not identify expert-parallel execution.' }
    return @{ message=@{role='assistant';content=[string]$answer.content}; response=$response; text=$responseText; seconds=$watch.Elapsed.TotalSeconds }
}

$models = Invoke-RestMethod "$base/models" -TimeoutSec 30
if (@($models.data | Where-Object id -EQ $model).Count -ne 1) { throw "Expected $model from $base/models." }
Write-Host 'Running two short requests: cold generation, then an appended turn.'
$messages = @(@{role='system';content="Smoke run $runId. You are a helpful assistant."},
    @{role='user';content='Say hello in one sentence. /no_think'})
$measurements = @()
$smokeStatus = 'failed'
try {
    foreach ($name in @('cold','append')) {
        $turn = Request-Turn $messages 128
        $turn.text | Set-Content -LiteralPath (Join-Path $results "$name.json") -Encoding utf8
        $headers = $turn.response.Headers
        $row = [ordered]@{ name=$name; wall_seconds=$turn.seconds }
        foreach ($field in @('Prompt-Tokens','Prefilled-Tokens','Reused-Tokens','Completion-Tokens','Prefill-TPS','Decode-TPS')) {
            if (!$headers.ContainsKey("X-Flash-Gordon-$field")) { throw "Missing timing header: $field" }
            $row[$field] = [double]::Parse([string]$headers["X-Flash-Gordon-$field"], [Globalization.CultureInfo]::InvariantCulture)
            if (![double]::IsFinite($row[$field]) -or $row[$field] -lt 0) { throw "Invalid timing header: $field" }
        }
        $row['prefix_cache'] = [string]$headers['X-Flash-Gordon-Prefix-Cache']
        $measurements += $row
        if ($row['Completion-Tokens'] -lt 1 -or $row['Completion-Tokens'] -gt 128) { throw 'Invalid generated-token count.' }
        if ($row['Prefilled-Tokens'] + $row['Reused-Tokens'] -ne $row['Prompt-Tokens']) { throw 'Inconsistent prompt accounting.' }
        if ($name -eq 'cold' -and ($row['Reused-Tokens'] -ne 0 -or $row.prefix_cache -ne 'miss')) { throw 'Cold smoke request unexpectedly reused context.' }
        if ($name -eq 'append' -and ($row.prefix_cache -ne 'hit' -or $row['Reused-Tokens'] -le 0 -or $row['Prefilled-Tokens'] -le 0)) { throw 'Append smoke request did not reuse its prefix.' }
        Write-Host ("{0}: {1} output tokens; prefill {2:N2} tok/s; decode {3:N2} tok/s; reused {4} tokens" -f $name,$row['Completion-Tokens'],$row['Prefill-TPS'],$row['Decode-TPS'],$row['Reused-Tokens'])
        Write-Host $turn.message.content
        if ($name -eq 'append' -and $turn.message.content -notmatch '(?i)\b(?:4|four)\b') { throw 'The arithmetic smoke check failed: expected four. Timing results and the actual reply were saved; generation speed is not a correctness pass.' }
        $messages += $turn.message
        $messages += @{role='user';content='What is two plus two? Answer briefly. /no_think'}
    }
    $smokeStatus = 'passed'
} finally {
    @{status=$smokeStatus; base_url=$base; run_id=$runId; checks=$measurements} |
        ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $results 'smoke.json') -Encoding utf8
    Write-Host "Results: $results`nAPI remains running at $base`nModel: $model"
}
Write-Host 'Smoke check passed. These short-context rates do not establish 128k performance.'
if (!$NoChat) {
    Write-Host 'Chat below. /clear starts fresh; /quit exits and leaves the server running.'
    $conversation = @(@{role='system';content='You are a helpful assistant.'})
    while ($true) {
        $prompt = Read-Host 'You'
        if ($prompt -eq '/quit') { break }
        if ($prompt -eq '/clear') { $conversation = @(@{role='system';content="You are a helpful assistant. Session $([guid]::NewGuid())."}); continue }
        if ([string]::IsNullOrWhiteSpace($prompt)) { continue }
        $candidate = $conversation + @(@{role='user';content=$prompt})
        try {
            Write-Host 'Generating...'
            $turn = Request-Turn $candidate 512
            Write-Host "`n$($turn.message.content)`n"
            $conversation = $candidate + @($turn.message)
        } catch { Write-Warning $_.Exception.Message }
    }
}
