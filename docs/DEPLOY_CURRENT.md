# Current production deployment

**Current selection: vision-text fix source `e66ff47` (production source
`7490482` plus the chat-render fix) on stack-1, sealed mix-04, sessions 2,
plus the ordered watchdog.** The prior lease authority explicitly
overrode the prior **0.965996x short-decode paired median**
(95% CI **[0.879970, 1.016834]**). This remains an **open performance item**,
not a statistical pass; no deployed binary/kernel change was made to hide it.

The preceding robustness release passed these functional gates: a 44-message approximately 44K agent retains
**44398 tokens** across a separate two-message title request;
worker loss fails the request while preserving the API PID and recovers;
coordinator SIGKILL triggers watchdog-only ordered restart. Next generations
succeed, all eight pinned identities match, and five real canonical greedy
outputs match production. Watchdog boot-enabled/active; memlock unlimited.

Production also misses loaded cold-32K chat QoS (8.37 TPS, p95 event gap
200.07 ms); the override does not claim that target is solved.
[Decision, functional receipts and open item](https://github.com/CuppaXanax/flash-gordon/blob/cuppaxanax/multi-session-int/results/hillclimb/MULTI_SESSION_OVERRIDE.md).

Vision-text promotion verified on **2026-10-20 UTC**: all eight live identities,
argv, FG environment, binary/manifest/shader hashes and unlimited process
memlock match the pinned selection. Health is `ok`; watchdog is enabled/active,
with `LimitMEMLOCK=infinity` and `KillMode=process`; no deadline timer is armed.
The exact single-message text-before-image password probe returns
`BANANA-SEVEN, red`; image-before-text returns `BANANA-SEVEN, Red`.
Both use **100 prompt tokens** for the 224×224 solid-red PNG fixture
(the ordering-equality gate passes; the historical fixture reported 101).
Image-only describes a solid red square, with 76 prompt tokens and a normal
stop after 300 completion tokens; its budget is 512 to allow reasoning.
The temperature-zero text probe returns exactly `12` before and after restart.
Two preliminary attempts rolled back successfully: an unnecessarily exact
101-token assertion, then an image-only 96-token budget that truncated reasoning.
Targeted `test_api` and `test_chat` pass; build flags are unchanged Makefile
defaults (`-O3 -g -std=c17`, warnings-as-errors), with shaders copied byte-for-byte
from the preceding immutable release, not rebuilt.
[Promotion and final identity receipts](../results/hillclimb/vision-text-20261020/verification.json).

| Item | Current value |
|---|---|
| Owners | 192.168.42.42–192.168.42.49; ranks 0–7 |
| API | `http://192.168.42.42:8080/v1` |
| Model | `Qwen3.8-Flash-Next-Uncensored` |
| Release | `/home/xander/flash-gordon-live/20261020-abq-vision-text` |
| Binary SHA256 | `686490e614a1296ea17a00871c1cf958542dae82b6d611882a274c79d71424c5` |
| Source | `e66ff47`; parent production source `7490482` |
| Shader-tree SHA256 | `5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633` |
| Manifest | `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm` |
| Manifest SHA256 | `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252` |
| Sessions / expert wide | 2 sessions; `FG_EXPERT_Q8_WIDE=1` |
| Retained operator | `/home/xander/flash-gordon-live/20261015-abq-operator` |

The preceding immutable release, weights, manifest and shader tree are unchanged.
Operator scripts, logs and PID records are outside that release. The same
`nohup` launch semantics as the original are retained. The enabled
`flash-gordon-ring.service` on .42 now provides ordered failure/boot recovery:
stop coordinator, stop workers, start workers, start coordinator. It uses the
same pinned `launch-current.sh`/`current.json`, dedicated restricted SSH and
strict host keys. Worker transport faults preserve the API while workers
reconnect; coordinator death triggers the full ordered restart.
The unit must retain `LimitMEMLOCK=infinity`: the default systemd allowance
caused `register fixed buffer: Cannot allocate memory` during the cold restart.
The initial baseline restoration reached `ring_ready` at 21:17:37 UTC after
correcting it. The preceding robustness candidate's SIGKILL restart is recorded above.
These were cold process restarts, not host reboots.

## Vision pack prerequisite

Vision requires **`<manifest directory>/tower.fgm`**, alongside `manifest.fgm`.
On the coordinator this is verified as
`/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/tower.fgm` ->
`/home/xander/fg-ring-pack-abliterated/tower.fgm`.
Future pack creation/copy procedures must include this tower file or a valid,
readable symlink to it; a manifest and rank weights alone are insufficient.
Preserve the symlink target when moving packs.

## Required supervision pause for every future fleet lease

Before any deployment, rollback or manual restart, run on .42:

```bash
op=/home/xander/flash-gordon-live/20261015-abq-operator
touch "$op/supervisor.pause"
sudo systemctl stop flash-gordon-ring.service
sudo systemctl is-active flash-gordon-ring.service  # must report inactive
```

Stopping this watchdog deliberately leaves serving processes alone
(`KillMode=process`). After restoring/pinning and verifying the selected ring:

```bash
rm -f "$op/supervisor.pause"
sudo systemctl start flash-gordon-ring.service
sudo systemctl is-enabled flash-gordon-ring.service
sudo systemctl is-active flash-gordon-ring.service
```

Do not leave a qualification deadline timer armed when releasing a fleet lease.
Restarting this unit alone does not restart an already healthy serving ring.

## Persistent launcher and exact commands

Each owner retains `launch-current.sh`, `launch-vision-text.sh`,
`ab-instance-vision-text.sh` and `current.json` in
the operator directory. The launcher pins the exact release/manifest, checks
the launcher body, binary, manifest and shader-tree hashes before start, and
derives its rank only from .42–.49.
The pinned instance script refuses duplicate serving processes. Stop enumerates owned
numeric PIDs and requires the exact executable and `--manifest` argument before
TERM/wait; there is no name kill, forced kill, or floating "latest" symlink.
Current all-eight identity/API receipts are linked above. Historical robustness
launcher and restart/identity/health receipts are in the
[override archive](https://github.com/CuppaXanax/flash-gordon/tree/cuppaxanax/multi-session-int/results/hillclimb/multi-session-override-20261018).
The earlier `prod-gap\current` and `ab-current` receipts are historical.

Run as `xander`. Coordinator command, from the immutable release directory:

```bash
ulimit -c 0
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20261020-abq-vision-text/vulkan \
  FG_EXPERT_Q8_WIDE=1 FG_RING_TRACE=1 \
  FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024 \
  FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 \
  FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2 \
  /home/xander/flash-gordon-live/20261020-abq-vision-text/flash-gordon api \
  --manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm \
  --model-name Qwen3.8-Flash-Next-Uncensored \
  --host 0.0.0.0 --port 8080 \
  --context-tokens 262144 --gpu-index-tokens 262144 \
  --qsa-hot-tokens 0 --qsa-page-cache-mib 700 --sessions 2 \
  >/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.log \
  2>&1 </dev/null &
echo "$!" >/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.pid
```

Workers use the following command with `N=1` on .43 through `N=7` on .49:

```bash
ulimit -c 0
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20261020-abq-vision-text/vulkan \
  FG_EXPERT_Q8_WIDE=1 \
  FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024 \
  FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 \
  FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2 \
  /home/xander/flash-gordon-live/20261020-abq-vision-text/flash-gordon rank \
  --manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm --rank "$N" \
  >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.log" \
  2>&1 </dev/null &
echo "$!" >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.pid"
```

Do not enable `FG_DECODE_PROFILE`; it disables static replay and changes the
measurement. No fusion/MTP or additional unlisted switches are part of this
deployment. The pinned script scrubs profiling/other experiment flags before
supplying the explicit recipe above. Only the coordinator receives `--sessions 2`. The QSA cache budget
is per session. Routing uses longest matching public-message prefixes, independent of the optional
`user` hint. Send exact assistant/tool transcripts for forward state reuse.
A system prompt alone is not affinity; shortened replays reset GDN state.

## Immediate rollback: preceding robustness release

Pause/stop supervision as above. Stop coordinator .42, then workers .43–.49
using the current pinned launcher. On all eight owners:

```bash
op=/home/xander/flash-gordon-live/20261015-abq-operator
cp "$op/launch-pre-vision-20261020.sh" "$op/launch-current.sh"
cp "$op/pre-vision-20261020.json" "$op/current.json"
```

Start workers 1–7 before coordinator 0 with
`bash "$op/launch-current.sh" start`, verify identities/health/real generation,
then resume supervision. This restores
`/home/xander/flash-gordon-live/20261018-abq-session-recovery-r2`, binary
`3405c8976eca580ca2e8690bb336b0ca13e69ab761dbd7bffee590bb280cad17`,
same mix-04, sessions 2, flags and shader hash. This exact rollback was exercised
twice during this lease; it reintroduces the text-before-vision rendering bug.
Do not overwrite these dated rollback files during later leases.

## Historical rollback: pre-robustness stack-1 baseline

Pause/stop supervision as above. Stop current coordinator, then workers.
On all eight owners:

```bash
op=/home/xander/flash-gordon-live/20261015-abq-operator
cp "$op/launch-pre-robustness-20261018.sh" "$op/launch-current.sh"
cp "$op/pre-robustness-20261018.json" "$op/current.json"
```

Start workers 1–7 before coordinator 0 with
`bash "$op/launch-current.sh" start`, verify real generation, then resume
supervision. This returns to immutable stack binary
`02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0`,
release `20261016-abq-stack-adaptive`, same mix-04 and sessions 2. It
reintroduces the old routing and listener-recovery defects.
Do not overwrite the dated rollback files during later leases.

The older pre-stack rollback below is historical, not the immediate target.
## Historical pre-stack adaptive rollback (sessions 2)

With the exclusive fleet lease:

```powershell
pwsh -NoProfile -File .\tools\ab-rollback-pre-stack.ps1
```

This stops the exact pinned current instance, restores retained
`launch-pre-stack-20261016.sh`/`pre-stack-20261016.json`, starts workers before
coordinator, and verifies all eight live executable/manifest/shader hashes,
argv, FG environment, model, health and real greedy generation.
It restores `2d0f8a7`, release `20261016-abq-multisess`, binary
`de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58`,
same mix-04, sessions 2 and wide 0. This actual stack-to-adaptive rollback
was exercised before the final full-stack re-promotion/restart.

## Older main/mix-04 fallback (sessions 1)

From this worktree, with the exclusive fleet lease:

```powershell
pwsh -NoProfile -File .\tools\ab-rollback-main-mix04.ps1
```

This uses retained `launch-main-mix04.sh`/`main-mix04.json`, stops owned numeric
PIDs through the current hash-pinned launcher, starts workers before coordinator,
updates local live state, and verifies health/models/real generation. It restores
`a63a166`, binary SHA256
`1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789`,
with the same sealed mix-04 manifest and no `--sessions` argument. The new
wrapper is syntax-checked; its A configuration was exercised repeatedly in the
paired study. The wrapper was not invoked after the final selected restart.

## Historical original-Q8 rollback (not the main/mix-04 fallback)

From the `cuppaxanax/ab-harness` worktree root, with the exclusive fleet lease:

```powershell
pwsh -NoProfile -File .\tools\ab-rollback-original.ps1
```

This was actually exercised before re-promoting mix-04. It stops the exact
current executable/manifest, restores retained `launch-original.sh` and
`original.json` as the pinned current selection on all eight, starts workers
before coordinator, and checks health/model/real greedy chat. It does not rely
on a name kill or on an ephemeral local record to identify the running process.
Original release and pack remain immutable:
`20260926-abl2/flash-gordon`, SHA256
`e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2`;
original manifest SHA256
`35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`.

## Restart the complete ring

Do not run this while another fleet operator holds the production lease.
Stop all eight, start the seven workers, then start .42. Use the retained
launcher rather than the historical name-kill scripts:

```powershell
$runner = Join-Path $PWD 'tools\ab-fleet.ps1'
$all = @('192.168.42.42','192.168.42.43','192.168.42.44','192.168.42.45',
         '192.168.42.46','192.168.42.47','192.168.42.48','192.168.42.49')
$workers = @('192.168.42.43','192.168.42.44','192.168.42.45','192.168.42.46',
             '192.168.42.47','192.168.42.48','192.168.42.49')
$null = New-Item -ItemType Directory -Force .build
$script = Join-Path $PWD '.build\restart-current.sh'
function Invoke-Current($action, $targets) {
    [IO.File]::WriteAllText($script,
        "bash '/home/xander/flash-gordon-live/20261015-abq-operator/launch-current.sh' $action`n")
    $r = @(& $runner -ScriptPath $script -Targets $targets -PassThru)
    if ($r.Count -ne $targets.Count -or @($r | Where-Object ExitCode -ne 0).Count) {
        throw "Current deployment $action failed; inspect owned-PID status."
    }
}
Invoke-Current stop $all
Invoke-Current start $workers
Invoke-Current start @('192.168.42.42')
```

Allow loading to finish, then verify `/health` reports `status=ok`,
`/v1/models` lists the exact model above, and a real greedy chat returns visible
content. `launch-current.sh status` reports process arguments and file hashes;
the final archived verification additionally checks all eight live executable
inodes, exact argv/environment and the pinned current identities.

Final verification on **2026-10-16 UTC-07**, after the complete selected ring
restart and observer cleanup, passed. All eight live executable/manifest hashes,
argv and allowed FG environment were checked. Health/models and real greedy
generation passed; a fresh genuine two-stream generation also passed after
restart. Only the flags above are present; no `LD_PRELOAD` or decode profiler.
[Final timestamp and API receipt](../results/hillclimb/stack-fleet-20261016/final-api-health.json);
the sibling `final-cleanup.json` verifies all-eight experiment-scratch cleanup.
Retained launchers survive logout; automatic host-boot/systemd persistence is
not claimed.
