# Bounded Qwen native-context qualification

This collector measures a real OpenAI-compatible session against a
representative coding corpus. It grows one append-only conversation to the
native context and takes three short continuation samples at each checkpoint.
It refuses repeated filler: use a repository, issue corpus, or other text that
resembles the intended Pi session.

Start an already-isolated candidate with the opt-in server summary file:

```sh
FG_QUALIFICATION_PROFILE=/tmp/qwen-server.jsonl ./flash-gordon-chat-api api \
  --manifest /path/to/candidate/manifest.fgm --host 0.0.0.0 --port 8080
python3 tools/qwen_native_qualification.py \
  --base-url http://127.0.0.1:8080/v1 --model Qwen3.8-Flash-Next \
  --corpus /path/to/representative-coding-corpus.txt \
  --output /tmp/qwen-client.jsonl --wall-budget-minutes 120 \
  --request-timeout-seconds 1800
```

Validate after the run:

```sh
python3 tools/validate_qwen_qualification.py /tmp/qwen-client.jsonl \
  --server-jsonl /tmp/qwen-server.jsonl --output /tmp/qwen-report.json
```

`FG_QUALIFICATION_PROFILE` is disabled by default. `1` writes
`fg-qualification-server.jsonl` in the server working directory; a non-empty
value other than `1`, `true`, `0`, or `false` is treated as the output path.
Each line is locked and flushed, and includes the server's actual token and
stage accounting. While this opt-in is active, the stream's final event also
carries standard OpenAI usage fields so the client can target checkpoints
without a tokenizer endpoint; ordinary streaming responses are unchanged.

The default thinking sampling policy is `temperature=1.0`, `top_p=0.95`, and
`top_k=20`; override it only for an explicitly different experiment. The
collector never deploys, quiesces, mutates LKG, hashes, repacks, or qualifies
multimodal behavior. It does not run a benchmark or fabricate token counts.

The native-context pass is bounded: it processes roughly 261,888 tokens once,
then performs only small decode tails at the checkpoints. Expected duration is
approximately native tokens divided by observed incremental prefill TPS, plus
the decode tails—not the sum of cold prefills at every checkpoint. A slow or
stalled request is bounded by the smaller of the remaining wall budget and the
per-request timeout. Progress messages include elapsed time and a clearly
labelled estimate based on the observed end-to-end session rate. The report shows actual context
and tolerance for each checkpoint; it never calls an overshoot an exact 4K
measurement.

The validator requires a server summary file, successful sampling records,
unique joined request IDs, consistent reused/prefilled accounting, monotonic
context, unique corpus ranges, dense curve points at 4K/8K/16K/32K/64K plus
131K/196K/261K native-context coverage, three decode samples per checkpoint,
and no unexplained adjacent drop over 35% or two-neighbor discontinuity over
2x in either decode or incremental prefill. Client TTFT begins at the first
non-empty generated content, reasoning, or tool-call delta; the role-only SSE
envelope does not count. The 600 prefill and 100 decode goals are reported as
aspirational values and are not failures. The enforced floors are 250 TPS for
incremental prefill near 4K and a 50 TPS decode median at every checkpoint.
