param([hashtable]$Deployment,[string]$Arm)
. 'D:\looking-glass-labs\fg-wt\ab-harness\tools\ab-stack-common.ps1'
Assert-ABStackLease 'D:\looking-glass-labs\fg-wt\ab-harness\.build\stack-live\lease.json'
& 'D:\looking-glass-labs\fg-wt\ab-harness\tools\ab-deploy.ps1' -Deployment $Deployment -Arm $Arm
