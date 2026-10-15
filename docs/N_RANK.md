# Runtime-configurable rank rings

## Contract and compatibility

The same executable supports N=4,6,8,12,16 (divisors of 48, with a minimum of
four and a maximum of sixteen). Runtime startup reads N from the sealed
manifest before opening models, validating routes, or making fabric connections.
`pack --ranks N` selects it when packing GGUF sources. The default remains 8.
There is one topology per process: changing N while a runtime or its transport
threads are live is unsupported. `FG_MAX_RANK_COUNT` is storage capacity;
`FG_RANK_COUNT` is the active process topology, not an array bound.

N=8 preserves existing placement, expert maps, head rank 4, seven canonical
n-gram ranges, arithmetic, payload sizes and subversions. v4/v5/v6 disk
manifests retain their original eight-rank ABI; expanded in-memory structs are
serialized explicitly. The golden test hashes were generated independently
from main at `1863c95`, not from the new serializer. Old binaries reject the new
v7 manifests. Non-eight manifests have fixed sixteen-rank storage capacity and
an explicit active rank count; unused entries are zero.

The outer transport protocol remains v6. The existing 16-byte prefill-result
header uses byte 12 for contributor bits 0-7 and, only for non-eight topologies,
byte 13 for bits 8-15. Its payload subversion is 4 rather than 3. Frame sizes,
float encodings, and the 40,960-byte hyper-connection vector do not change.
Manifest-hash handshakes prevent mixing rings or old/new placement.

For N other than 8, layer `l` belongs to `l / (48/N)`. All 512 experts and dense
weights of a layer are local to its owner. N=12 gives:

| Rank | Blade suffix | Layers | GDN / QSA |
|---:|---:|---|---|
| 0 | .42 | 0-3 | 3 / 1; PLE and embedding |
| 1 | .43 | 4-7 | 3 / 1 |
| 2 | .44 | 8-11 | 3 / 1 |
| 3 | .45 | 12-15 | 3 / 1 |
| 4 | .46 | 16-19 | 3 / 1 |
| 5 | .47 | 20-23 | 3 / 1 |
| 6 | .48 | 24-27 | 3 / 1 |
| 7 | .49 | 28-31 | 3 / 1 |
| 8 | .50 | 32-35 | 3 / 1 |
| 9 | .51 | 36-39 | 3 / 1 |
| 10 | .52 | 40-43 | 3 / 1 |
| 11 | .53 | 44-47 | 3 / 1; output bundle |

Explicit non-eight expert maps must remain owner-local. This change does not
introduce arbitrary-width expert collectives or change top-10 routing.

Rank 0 launches its own first stage through a local inbox, not a fabric
self-send, for prefill, single-token decode and depth-B decode. Layer 1
receives its PLE n-gram tensor on that stage. Local-first prefill limits
in-flight chunks to the smaller of eight and N-1, so a blocked final-stage
return cannot exhaust the downstream stages while the coordinator issues.
The existing remote-first eight-rank scheduling is unchanged. The CPU-only
runtime inbox regression can be run with:

```sh
make tests/test_prefill_collective
./tests/test_prefill_collective --local-ring-launch
```

## Head choice and network cost

The non-eight baseline puts the complete output bundle on the last stage.
It uses the existing full-head sampler for greedy and non-greedy decoding.
It does not load helper vocabulary slices or duplicate the output bundle in
the coordinator. The eight-rank four-way head remains unchanged.

Four-way splitting is attractive for greedy latency, but raises helper memory
use and its existing direct-handoff machinery requires a distinct final-stage
and head owner. Rather than silently running that machinery with self-sends,
non-eight rings use the supported coordinator relay: last stage -> coordinator
-> last-stage full head. The ledger therefore reports **13 full-vector sends**
for the twelve-rank decode path (11 stage boundaries plus 2 relay sends).
That is 4.75 ms of payload serialization at 112 MB/s, excluding headers,
turnaround, n-gram requests and the small sampled-token return. This is not
the hypothetical eleven-send, colocated-head roofline in DESIGN section 3.
Fusing the local final-stage/head handoff and then measuring an optional
distributed head is follow-up work, not a claimed result here.

## Lossless repacking without the deleted GGUF

`repack` reads a **local assembled pack**, not blades or GGUF files:

* `manifest.fgm`: the sealed manifest actually used by the source deployment;
* every referenced `rank-XX.fgw` owner-local weight file;
* `tokenizer/tokenizer.fgt`;
* `ngram.iq4nl`, or all sealed `ngram-rank-XX.iq4nl` source shards.

Copy the active source manifest to the local name `manifest.fgm` if its deployed
filename is different. Do not substitute an unsealed historical manifest.
The source pack must have all 512 experts of each layer in one record per
family. Split expert packs, host-cache embeddings, and combined vision packs
are rejected rather than guessed; separate vision-tower packs are unchanged.

Each tensor extent is SHA-256 checked while copied. Cooked Q8, K-quant and
Q5 expert layouts remain cooked, in original expert order. No dequantization,
requantization, quality trade or dependency on the missing source GGUF occurs.
Source identity and quantization-profile hashes are retained. If only n-gram
shards exist, they are concatenated in sealed row order and the full-table
hash is checked before splitting into the new ranges:

```
begin(r) = floor(320001536 * (r-1) / (N-1))
end(r)   = floor(320001536 * r / (N-1)), r=1..N-1
```

N=8 is the exception: its historical, unequal ranges remain exact.
Repack writes to a new directory, refuses existing output, removes its own
partial artifacts on failure, and publishes a freshly sealed manifest.
`memory.json` records exact per-rank byte counts from that output.
The input directory is read-only. For N=8 -> N=8, source ownership is retained.

The historical `/home/xander/abliterated-recon/pack-run.sh` was not available
locally and was **not fetched from the fleet**. Its behavior is not assumed.
The semantics above come from this repository's packer, manifest, loader and
quant-layout code. There were no fleet reads or writes in this task.

### Exact local repack commands

Run on a Linux workstation with the assembled source pack. On this Windows
checkout, the same build/tests run through Ubuntu-24.04 WSL. Keep compiler
scratch files under the checkout:

```bash
mkdir -p .build
export TMPDIR="$PWD/.build"
make -j8 flash-gordon
make test-topology
./flash-gordon repack \
  --input /home/xander/fg-ring-pack-abliterated-local \
  --output /home/xander/fg-ring-pack-abliterated-12 \
  --ranks 12 --endpoint-prefix 192.168.42. --endpoint-first 42
./flash-gordon inspect \
  --manifest /home/xander/fg-ring-pack-abliterated-12/manifest.fgm
python3 -c 'import json; p=json.load(open("/home/xander/fg-ring-pack-abliterated-12/memory.json")); print(json.dumps(p,indent=2)); assert all(r["pageable_estimate"]<=8000000000 for r in p["ranks"]), "8 GB planning gate failed; do not deploy on that budget"'
```

The planning assertion can fail for the current pack; do not remove it merely
to claim a fit. See the ledger below. If authorized GGUF sources still exist,
`pack --ranks 12 --output NEW --source ... --source-sha256 ...` supports the same
topology. The existing source packer's model/source identity checks still
apply. Run `repack --input NEW --output SEALED --ranks 12 ...` to materialize
and seal resident n-gram shards before deployment.

## Per-rank memory ledger

**Eight GB is not a demonstrated fit.** The table is a historical-schema
estimate, not a measurement of the abliterated deployment. It retains Q8 dense
weights, the documented mixed expert formats and exceptions at layers
2/4/30/46/47, and 4-KiB tensor alignment. MTP, tokenizer/host overhead and
additional concurrent-session state are not included. `memory.json` from the
actual repack supersedes this estimate, including any source MTP tensors.

Common to every rank at the native 262144-token, microbatch-128, window-2
profile: transient reserve **536,870,912 B**, driver reserve **536,870,912 B**,
KV/index/GDN ledger **48,726,016 B**, and QSA state-file disk allocation
**536,879,104 B** (not permanently resident). Scratch is **335,544,320 B** on
rank 0 and **201,326,592 B** elsewhere. The N=12 model has one QSA per rank,
so scratch and session accounting are derived from ownership, not rank 3/7.

All values below are **bytes**, not GiB. Persistent includes embedding on
rank 0 and the complete output bundle on rank 11.

| Rank | Persistent weights | N-gram mapped span | Conservative total | Pageable planning estimate |
|---:|---:|---:|---:|---:|
| 0 | 7,816,622,080 | 0 | 9,274,634,240 | 9,274,634,240 |
| 1 | 6,896,373,760 | 2,618,194,320 | 10,838,362,512 | 8,354,385,920 |
| 2 | 6,634,229,760 | 2,618,194,410 | 10,576,218,602 | 8,092,241,920 |
| 3 | 6,634,229,760 | 2,618,194,410 | 10,576,218,602 | 8,092,241,920 |
| 4 | 6,634,229,760 | 2,618,194,320 | 10,576,218,512 | 8,092,241,920 |
| 5 | 6,634,229,760 | 2,618,194,410 | 10,576,218,602 | 8,092,241,920 |
| 6 | 6,634,229,760 | 2,618,194,410 | 10,576,218,602 | 8,092,241,920 |
| 7 | 6,896,373,760 | 2,618,194,410 | 10,838,362,602 | 8,354,385,920 |
| 8 | 6,634,229,760 | 2,618,194,320 | 10,576,218,512 | 8,092,241,920 |
| 9 | 6,634,229,760 | 2,618,194,410 | 10,576,218,602 | 8,092,241,920 |
| 10 | 6,634,229,760 | 2,618,194,410 | 10,576,218,602 | 8,092,241,920 |
| 11 | 7,840,952,320 | 2,618,194,410 | 11,782,941,162 | 9,298,964,480 |

Conservative total = persistent + complete mapped n-gram span + transient +
KV + scratch + driver. Pageable estimate substitutes the existing 128-MiB
n-gram hot region for the complete mapped span; it is **not a hard RSS bound**.
Unpinned page cache is reclaimable, and cache misses can hurt latency.
`FG_NGRAM_PAGEABLE=0` requests full pinning and must be budgeted conservatively.
The maximum conservative estimate is about 10.97 GiB, below the historical
13.5-GiB cap but above the requested ~8-GB usable budget. Even an 8-GiB
interpretation does not fit the estimated first/last stages. More blades
increase headroom versus six-layer stages, but do not make those endpoint
costs disappear. Q4 dense, revised reservations, pageable embedding or head
rebalancing need their own validated ledger; this topology change does not
silently alter precision or lower safety reserves.

## Twelve-blade deployment commands (future, not executed here)

Acquire the fleet lease and an approved maintenance window first. Do not
run alongside the old eight-rank runtime: memory and TCP port 19100 conflict.
Complete the memory gate above, retain the old binary/pack/launch configuration,
then stop the old deployment through its existing operator-controlled launch
mechanism. Do not infer PIDs or kill processes by name.

After those prerequisites, these Linux/SSH commands stage into a separate
directory and start ranks .42-.53. They never overwrite the production pack.
SSH key authentication and writable `/home/xander` are prerequisites.
Build shaders on the workstation with the repository's existing compiler
wrapper; no shader changes are needed for topology. This workstation binary
must use a glibc/Vulkan ABI compatible with the blades; otherwise use the
remote-build PowerShell helper below after the same maintenance prerequisites.

```bash
mkdir -p .build
make -j8 all
PACK=/home/xander/fg-ring-pack-abliterated-12
RELEASE=/home/xander/fg-n-rank-12
test -f "$PACK/memory.json"
tar -czf .build/n-rank-runtime.tar.gz flash-gordon vulkan
for rank in $(seq 0 11); do
  host="xander@192.168.42.$((42+rank))"
  ssh "$host" "test ! -e '$RELEASE'; test ! -e '$PACK'; mkdir '$RELEASE' '$PACK'"
  scp .build/n-rank-runtime.tar.gz "$host:$RELEASE/runtime.tar.gz"
  scp "$PACK/manifest.fgm" "$PACK/memory.json" "$host:$PACK/"
  printf -v weight 'rank-%02d.fgw' "$rank"
  scp "$PACK/$weight" "$host:$PACK/"
  if [ "$rank" -eq 0 ]; then
    scp "$PACK/ngram.iq4nl" "$host:$PACK/"
    scp -r "$PACK/tokenizer" "$host:$PACK/"
  else
    printf -v shard 'ngram-rank-%02d.iq4nl' "$rank"
    scp "$PACK/$shard" "$host:$PACK/"
  fi
  ssh "$host" "cd '$RELEASE'; tar -xzf runtime.tar.gz; chmod +x flash-gordon; ./flash-gordon inspect --manifest '$PACK/manifest.fgm'"
done
# The previous deployment must already be stopped on every old blade.
for rank in $(seq 1 11); do
  ssh "xander@192.168.42.$((42+rank))" \
    "cd '$RELEASE'; FG_NGRAM_PAGEABLE=1 nohup ./flash-gordon rank --manifest '$PACK/manifest.fgm' --rank $rank >server.log 2>&1 </dev/null & echo \$! >server.pid"
done
ssh xander@192.168.42.42 \
  "cd '$RELEASE'; FG_NGRAM_PAGEABLE=1 nohup ./flash-gordon api --manifest '$PACK/manifest.fgm' --model-name Qwen3.8-Flash-Next-Uncensored --host 0.0.0.0 --port 8080 >server.log 2>&1 </dev/null & echo \$! >server.pid"
curl --fail --retry 60 --retry-connrefused --retry-delay 5 \
  http://192.168.42.42:8080/v1/models
```

On Windows, the existing deployment wrapper also accepts `-RankCount 12` and
checks the downloaded manifest count before stopping/starting its own release.
It assumes weight artifacts have already been staged:

```powershell
$env:FG_FLEET_PREFIX = '192.168.42.'
$env:FG_BASE_URL = 'http://192.168.42.42:8080/v1'
.\tools\start-smoke-chat.ps1 -RankCount 12 -SshUser xander `
  -Model Qwen3.8-Flash-Next-Uncensored `
  -PackDirectory /home/xander/fg-ring-pack-abliterated-12 `
  -ManifestName manifest.fgm -RemoteDirectory /home/xander/fg-n-rank-live -NoChat
.\tools\pi-stability.ps1 -RankCount 12 -Model Qwen3.8-Flash-Next-Uncensored `
  -ApiUrl http://192.168.42.42:8080/v1/chat/completions `
  -FleetRunner D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1
```

The historical TPS thresholds are not twelve-rank measurements; retain them
as explicit qualification gates or predeclare replacement thresholds. The
startup ledger must report `ranks=12`, `blocks=0:4,1:4,...,11:4`, and
`wire_hops=13`. Profile analysis accepts `--ranks 12`; default eight-rank
analysis remains unchanged. Historical shape-rate tables remain historical,
not twelve-rank placement ledgers.

Before accepting the deployment, run the existing fixed-prompt token/logit
regressions against N=8, reference/Vulkan/long-context gates, and client-wall
TTFT/TPS checks. This task verified local C17 compilation/linking, core tests,
golden ABI identity, topology/manifest/mask and repack fixtures, not GPU
numerical identity on production weights or fleet throughput. Roll back by
stopping only PIDs from the new release's `server.pid` files and restarting
the saved eight-rank launch configuration against the untouched old pack.

The full localhost `test_prefill_collective` currently fails scenario 0 with
`invalid expert prefill arguments`; the same failure was reproduced from
unmodified main at `1863c95`. Its new `--local-ring-launch` regression passes
independently. This work does not claim a passing full collective suite.
