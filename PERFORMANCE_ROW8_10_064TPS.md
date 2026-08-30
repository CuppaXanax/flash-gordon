# Cooked-Q8 Row-8 Decode Selection

## Result

The sealed decode path now selects an unsplit row-8 PSO for the measured cooked-Q8 shapes, while `2560 -> 6144` remains on row-4. Prefill remains unchanged.

- Final-20 frame mean: **99.3628 ms**, or **10.06413 TPS**.
- Final-20 48-layer mean: **94.9939 ms**.
- Response text, generated token IDs, and logits exactly match the accepted LKG oracle.
- The selection adds no dispatches, reductions, scratch buffers, or runtime tuning controls.
- Per-shape split variants remain rejected after token divergence and a `100.39115 ms` fleet result.
- Q5_1 Expert Supertile v2 remains rejected after its `100.8844 ms` fleet result.

The result improves the accepted `99.64675 ms / 10.03545 TPS` reference by `0.28395 ms`, but does not establish enough independent margin for a new LKG tag. `lkg-10.035tps-cooked-experts` remains the qualified reference.

## Selection

Row-8 is used only for single-token cooked-Q8 decode at these shapes:

- `320 -> 10240`
- `2560 -> 10240`
- `6144 -> 2560`
- `2560 -> 640`
- `640 -> 2560`
- `2560 -> 2560`
- `2560 -> 12288`
- `2560 -> 512`

All other shapes, including `2560 -> 6144`, continue to use the existing row-4 PSO.