$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $root
$output=Join-Path $root 'results\hillclimb\multisess-fleet-20261016'
Write-Host "WAITING for all ten complete paired batteries $([DateTimeOffset]::Now.ToString('o'))"
while($true){
    $count=0
    foreach($pair in 0..4){
        foreach($arm in @('A','B')){
            $path="$output\pair-$pair-$arm\battery.json"
            if(Test-Path $path){
                try{if((Get-Content $path -Raw|ConvertFrom-Json).complete){$count++}}catch{}
            }
        }
    }
    if($count -eq 10){break}
    if([DateTimeOffset]::Now -gt [DateTimeOffset]::Parse('2026-10-16T01:40:00-07:00')){
        throw "Pair deadline reached: $count/10 complete; do not claim five pairs."
    }
    Start-Sleep -Seconds 15
}
Write-Host "PAIRS COMPLETE; analysis/capacity $([DateTimeOffset]::Now.ToString('o'))"
python .build\analyze-multisess-fleet.py *> "$output\analysis.log"
if($LASTEXITCODE -ne 0){throw 'Five-pair analysis failed'}
& pwsh -NoProfile -File .build\multisess-capacity.ps1 *> "$output\capacity-driver.log"
$capacityExit=$LASTEXITCODE
python .build\analyze-multisess-memory.py *> "$output\capacity\memory-analysis.log"
$memoryExit=$LASTEXITCODE
$analysis=Get-Content "$output\analysis.json" -Raw|ConvertFrom-Json
$capacity=Get-Content "$output\capacity\capacity.json" -Raw|ConvertFrom-Json
$promote=$capacityExit -eq 0 -and $memoryExit -eq 0 -and $analysis.protected_pass -and
    $analysis.aggregate_improves -and $analysis.aggregate_clear_win -and $analysis.solo_hash_identity -and
    $analysis.concurrent_hash_identity -and $capacity.complete -and $capacity.phase1.correctness_pass
if($promote){
    $memory=Get-Content "$output\capacity\memory-summary.json" -Raw|ConvertFrom-Json
    $promote=$memory.safe
}
$arm=if($promote){'B'}else{'A'}
Write-Host "PINNING selected=$arm $([DateTimeOffset]::Now.ToString('o'))"
& pwsh -NoProfile -File .build\pin-multisess-selection.ps1 -Arm $arm *> "$output\selection-driver.log"
if($LASTEXITCODE -ne 0){throw 'Persistent selection failed; inspect restoration receipts'}
Write-Host "SELECTION COMPLETE arm=$arm $([DateTimeOffset]::Now.ToString('o'))"
