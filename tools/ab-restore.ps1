#requires -Version 7.0
$original = @{
    release = '/home/xander/flash-gordon-live/20260926-abl2'
    manifest = '/home/xander/fg-ring-pack-abliterated/manifest.fgm'
    manifest_sha256 = '35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb'
    binary_sha256 = 'e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2'
    sessions = 1; wide = 0
}
& (Join-Path $PSScriptRoot 'ab-deploy.ps1') -Deployment $original -Arm restore
if (-not $?) { throw 'Original restore failed.' }
