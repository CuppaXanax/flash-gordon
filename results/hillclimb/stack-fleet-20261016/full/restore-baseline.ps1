$c=Get-Content 'D:\looking-glass-labs\fg-wt\ab-harness\.build\stack-live\full\baseline.json' -Raw|ConvertFrom-Json -AsHashtable
& 'D:\looking-glass-labs\fg-wt\ab-harness\tools\ab-deploy.ps1' -Deployment $c -Arm restore-baseline
