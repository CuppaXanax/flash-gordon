param([switch]$Resume)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path
$fleet = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
function Invoke-Owned($name, $script, $upload = '', $destination = '') {
    $parameters = @{
        Targets = @('192.168.42.50')
        ScriptPath = (Join-Path $root "tools\$script")
        PassThru = $true
        CommandTimeout = 86400
        Sudo = $true
    }
    if ($upload) {
        $parameters.UploadPath = Join-Path $root $upload
        $parameters.UploadDestination = "/home/xander/hc-reconcile/$destination"
    }
    $result = & $fleet @parameters
    $result | ConvertTo-Json -Depth 20 | Set-Content ".build\reconcile\$name.json"
    if ($result.ExitCode -ne 0) { throw "$name failed: $($result.Stderr)" }
    Write-Output "$name COMPLETE"
}
Invoke-Owned 'controls-install' 'reconcile-install.sh' '.build\reconcile\controls.tar' 'controls.tar'
if (-not $Resume) {
    Invoke-Owned 'rank0-upload' 'reconcile-probe.sh' '.build\retained-pack\pack\rank-00.fgw' 'pack/rank-00.mixed.fgw'
}
foreach ($rank in @(1, 0, 2, 3, 4, 5, 6, 7)) {
    $upload = ''
    $destination = ''
    if ($rank -ne 0 -and -not ($Resume -and $rank -eq 1)) {
        $number = '{0:00}' -f $rank
        $upload = ".build\retained-pack\pack\rank-$number.fgw"
        $destination = "pack/rank-$number.mixed.fgw"
    }
    Invoke-Owned "rank$rank-crossrun" 'reconcile-step.sh' $upload $destination
}
foreach ($kind in @('evidence', 'logits')) {
    $result = & $fleet -Targets @('192.168.42.50') -ScriptPath "$root\tools\reconcile-status.sh" `
        -DownloadSource "/home/xander/hc-reconcile/reconcile-$kind.tar.gz" `
        -DownloadPath "$root\.build\reconcile\$kind.tar.gz" -CommandTimeout 86400 -PassThru -Sudo
    $result | ConvertTo-Json -Depth 20 | Set-Content ".build\reconcile\$kind-download.json"
    if ($result.ExitCode -ne 0) { throw "$kind download failed" }
}
Write-Output 'SIX-POLICY 560-CASE CROSSRUN COMPLETE; exported all evidence and logits'
