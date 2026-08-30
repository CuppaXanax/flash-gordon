---
name: flash-gordon-fleet
description: "Manage the Flash Gordon BC250 fleet. Use when deploying, fingerprinting, starting, benchmarking, stopping, or troubleshooting the eight-blade LKG, and before testing HTTP or chat serving candidates."
---

# Flash Gordon Fleet

## Invariants

- Run every blade operation through `D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1`. Never use raw `ssh` or `scp`.
- The active fleet is `.42` through `.49`; rank is the final octet minus 42. Rank 0 is the coordinator.
- Keep `lkg-10.035tps-cooked-experts` and `/home/xander/flash-gordon-q38-cooked/flash-gordon-expert-cooked` intact. Build serving experiments under separate names.
- Start workers without sudo. Run the coordinator with `-Sudo` because its mapped arena can otherwise fail under GTT pressure.
- The France prompt is an exact response/token oracle only, never placement input.

## Run The LKG Benchmark

From PowerShell:

```powershell
Set-Location D:\looking-glass-labs\bc-250-dbg
$fleet = 42..49 | ForEach-Object { "192.168.42.$_" }
$workers = 43..49 | ForEach-Object { "192.168.42.$_" }
$stateRanks = @("192.168.42.42", "192.168.42.45", "192.168.42.49")

.\Invoke-BC250Fleet.ps1 .\stop-fg-expert-cooked.tmp.sh -Targets $fleet -Sudo
.\Invoke-BC250Fleet.ps1 .\clean-qsa-state.sh -Targets $stateRanks -Sudo
.\Invoke-BC250Fleet.ps1 .\check-fg-expert-cooked-fingerprint.tmp.sh -Targets $fleet
.\Invoke-BC250Fleet.ps1 .\start-fg-expert-cooked-worker.tmp.sh -Targets $workers
.\Invoke-BC250Fleet.ps1 .\eval-fg-expert-cooked.tmp.sh -Targets 192.168.42.42 -Sudo
.\Invoke-BC250Fleet.ps1 .\collect-fg-expert-cooked-final.tmp.sh -Targets 192.168.42.42
```

Require 8/8 matching fingerprints before launch. Qualify decode with the unprofiled final 20 frame mean; token-30 profiling is diagnostic only. Stop the fleet after a run and remove transient `session-*.qsa` files from the actual candidate root.

## Deploy Or Serve A Candidate

1. Pin the candidate to an explicit commit and apply any uncommitted experiment as `-PatchPath`; do not mutate the installed LKG.
2. Install a separately named binary and shader directory on all eight blades, then verify one binary hash and one complete SPIR-V hash across 8/8 ranks.
3. Launch rank workers first, then the sudo coordinator. Use a dedicated helper script for any long-running HTTP/chat process and invoke it through `Invoke-BC250Fleet.ps1`.
4. Keep fabric port `19100` reserved for rank traffic. Choose and document a separate serving port.
5. Before comparing endpoint latency, rerun the LKG benchmark above to confirm fleet health and preserve exact greedy behavior.

HTTP serving currently fails closed until its request path is implemented and qualified. Do not treat process startup, a mock response, or a kernel-only benchmark as serving validation.