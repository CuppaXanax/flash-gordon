function Get-ABStackFlagNames {
    @('FG_ROOFLINE','FG_ROOF_DENSE','FG_ROOF_QSA','FG_ROOF_EXPERT',
      'FG_ROOF_HC','FG_ROOF_RECURRENT','FG_ROOF_GDN',
      'FG_NGRAM_LOCAL_CACHE','FG_EXPERT_Q8_WIDE','FG_DECODE_SUBMIT_GROUP')
}

function Get-ABStackFlags {
    param([System.Collections.IDictionary]$Deployment)
    $flags = [ordered]@{ FG_EXPERT_Q8_WIDE = [string]$Deployment.wide }
    if ($flags.FG_EXPERT_Q8_WIDE -notin @('0','1')) { throw 'wide must be 0 or 1.' }
    if ($Deployment.Contains('flags')) {
        foreach ($name in @($Deployment.flags.Keys | Sort-Object)) {
            $value = [string]$Deployment.flags[$name]
            if ($name -cnotin (Get-ABStackFlagNames) -or $value -notmatch '^(0|[1-9][0-9]{0,9})$' -or
                [uint64]$value -gt [uint32]::MaxValue) { throw "Invalid stack flag $name=$value" }
            if ($name -in @('FG_NGRAM_LOCAL_CACHE','FG_EXPERT_Q8_WIDE') -and $value -notin @('0','1')) {
                throw "$name must be 0 or 1."
            }
            if ($name -eq 'FG_DECODE_SUBMIT_GROUP' -and $value -notin @('0','1','2')) {
                throw 'FG_DECODE_SUBMIT_GROUP must be 0, 1 or 2.'
            }
            if ($name -eq 'FG_EXPERT_Q8_WIDE' -and $value -ne [string]$Deployment.wide) {
                throw 'flags.FG_EXPERT_Q8_WIDE disagrees with wide.'
            }
            $flags[$name] = $value
        }
    }
    return $flags
}

function Assert-ABDeployment {
    param([System.Collections.IDictionary]$Deployment, [switch]$Capture)
    if ($Deployment.release -ne '/home/xander/flash-gordon-live/20260926-abl2' -and
        $Deployment.release -notmatch '^/home/xander/flash-gordon-live/[0-9]{8}-abq[-a-zA-Z0-9]+$') {
        throw 'Deployment release must be original or an isolated YYYYMMDD-abq release.'
    }
    if ($Deployment.manifest -notin @('/home/xander/fg-ring-pack-abliterated/manifest.fgm',
        '/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm') -or
        [string]$Deployment.sessions -notin @('1','2') -or
        $Deployment.binary_sha256 -notmatch '^[a-f0-9]{64}$' -or
        $Deployment.manifest_sha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid pinned deployment identity.' }
    $null = Get-ABStackFlags $Deployment
    if ($Deployment.Contains('shader_tree_sha256') -and
        $Deployment.shader_tree_sha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid shader-tree digest.' }
    if ($Capture -and ($Deployment.capture_program -notmatch '^/home/xander/flash-gordon-live/[0-9]{8}-abq[-a-zA-Z0-9]+/q4-eval-batch$' -or
        $Deployment.capture_sha256 -notmatch '^[a-f0-9]{64}$')) { throw 'Pin a source-matched quality helper.' }
}

function Get-ABShaderTreeCommand {
    param([System.Collections.IDictionary]$Deployment)
    Assert-ABDeployment $Deployment
    return @"
set -euo pipefail
cd '$($Deployment.release)'
test -d vulkan && test ! -L vulkan
test -z "`$(find vulkan -type l -print -quit)"
find vulkan -type f -name '*.spv' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d ' ' -f 1
"@
}

function Get-ABDeploymentIdentity {
    param([System.Collections.IDictionary]$Deployment)
    $flags = Get-ABStackFlags $Deployment
    $sorted = [ordered]@{}
    foreach ($key in @($flags.Keys | Sort-Object)) { $sorted[$key] = $flags[$key] }
    $identity = [ordered]@{
        release=$Deployment.release; binary_sha256=$Deployment.binary_sha256
        manifest=$Deployment.manifest; manifest_sha256=$Deployment.manifest_sha256
        sessions=$Deployment.sessions; flags=$sorted
        shader_tree_sha256=$Deployment['shader_tree_sha256']
        capture_program=$Deployment['capture_program']; capture_sha256=$Deployment['capture_sha256']
    } | ConvertTo-Json -Depth 15 -Compress
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
        [Text.Encoding]::UTF8.GetBytes($identity))).ToLowerInvariant()
}

function Get-ABKernelIdentity {
    param([System.Collections.IDictionary]$Deployment)
    $copy = [ordered]@{}
    foreach ($key in $Deployment.Keys) { $copy[$key] = $Deployment[$key] }
    $copy.flags = Get-ABStackFlags $Deployment
    $copy.flags.Remove('FG_DECODE_SUBMIT_GROUP')
    Get-ABDeploymentIdentity $copy
}

function Assert-ABStackLease {
    param([string]$Path)
    $lease = Get-Content $Path -Raw | ConvertFrom-Json -AsHashtable
    $expected = @(42..49 | ForEach-Object { "192.168.42.$_" })
    if ($lease.multisession_released -isnot [bool] -or $lease.mtp_released -isnot [bool] -or
        -not $lease.multisession_released -or -not $lease.mtp_released -or
        @($lease.targets).Count -ne 8 -or
        ((@($lease.targets | Sort-Object -Unique) -join ',') -ne ($expected -join ',')) -or
        [DateTimeOffset]::Parse($lease.expires_at) -le [DateTimeOffset]::Now) {
        throw 'Stack execution requires a current exclusive .42-.49 lease after both prior leases release.'
    }
}
