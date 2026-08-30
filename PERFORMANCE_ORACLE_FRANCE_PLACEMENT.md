# France-Derived Expert Placement Oracle

## Qualification Status

**DIAGNOSTIC ORACLE ONLY. NON-QUALIFYING. DO NOT PROMOTE, TAG, SHIP, OR COUNT AS VERIFIED TPS.**

The placement map was trained on routes emitted by the France qualification prompt. The prompt is test data, not placement-training data. The map payload is deliberately absent from this repository; only its identity and measured diagnostic result are retained.

Round-robin remains production default behavior. `--expert-map` is an optional pack-time input, and the sealed runtime does not read or require an expert-map file.

## Identity

- Date: 2026-08-30
- Base LKG: `cb2ad0b5d9e0883a53dfa964fa89813d2ce28555` / `lkg-7.924tps-paired-q8`
- Manifest SHA-256: `f57cfc088827f69ada43f3e7a9378d09be0984aa0e81e0287afd8251f9c78975`
- Binary SHA-256: `1450d97d5dcf799916e239c31fe619f9b6e5de6e36ab9b07e9603a7d7b628370`
- Oracle map SHA-256: `38f25f640aad6959c78694bb9f86d0b93aec788cdd192323e52b34fbfc0a4cec`
- Eight-blade homogeneous fingerprints: PASS
- Source replay: 1,655 pass, 0 fail
- Fleet-side marker: `ORACLE_ONLY=1`, `PROMOTION_ALLOWED=0`

## Result

| Measurement | Qualified LKG | Oracle | Delta |
|---|---:|---:|---:|
| Final-20 mean TPS | 7.924 | 8.351 | +0.427 / +5.39% |
| Final-20 mean frame | 126.200 ms | 119.750 ms | -6.450 ms |
| Final-20 median frame | 126.000 ms | 120.000 ms | -6.000 ms |
| Stage-trace mean frame | 126.129 ms | 119.656 ms | -6.473 ms |
| Stage-trace layer graph | 121.753 ms | 115.301 ms | -6.452 ms |
| Token-30 collect | 59.726 ms | 47.973 ms | -11.753 ms / -19.68% |
| Token-30 collect per layer | 1.244 ms | 0.999 ms | -0.245 ms |
| Token-30 sync1 | 45.298 ms | 45.596 ms | +0.298 ms |
| Token-30 layer rows | 127.932 ms | 124.889 ms | -3.043 ms |

Final-20 frame range: 119.000-121.000 ms.

The token-30 fire comparison is not clean because the oracle runtime prints all ten expert IDs and owner ranks inside the profiled fire path, while the accepted trace used the shorter route record. Steady tail frames do not enable that instrumentation.

## Correctness

- Exact France response: PASS
- Retained token overlap: 20/20 exact
- Complete 48-layer expert-parallel trace: PASS
- Runtime ownership versus oracle map: PASS
- Busiest selected rank: three experts on 47 layers, four on one layer

Known former straggler collect times:

| Layer | Collect | Slow selected |
|---:|---:|---:|
| 22 | 0.952 ms | 3 |
| 23 | 1.069 ms | 3 |
| 31 | 1.079 ms | 3 |
| 34 | 0.961 ms | 3 |

The complete layer table is retained in [PERFORMANCE_ORACLE_FRANCE_PLACEMENT_LAYERS.csv](PERFORMANCE_ORACLE_FRANCE_PLACEMENT_LAYERS.csv).

## Decision

Clairvoyant placement removes 11.753 ms from profiled collect but leaves 47.973 ms of collect and a 119.750 ms frame. Placement is meaningful but insufficient for the 100 ms gate. No calibration-corpus map is being built in the next workstream.

The active target is the workload-independent N-selected-expert worker primitive: BC250-native routed weight layouts, multi-expert scheduling, activation reuse, and a resident graph through result reduction/return. The immediate collect goal is at most 0.90 ms/layer average.
