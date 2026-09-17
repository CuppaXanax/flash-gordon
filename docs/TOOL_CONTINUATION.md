# Tool metadata and prefix continuation

Date: 2026-09-17. Status: implemented on `fix/tool-churn-continuation`
(base `0a76474`, evidence snapshot `bebc16f`), local gates green, fleet A/B
pending.

## Problem

The OpenAI-compatible API reuses the runtime token prefix across turns
(`api_public_session_prefix`, `src/api.c`). Before this change the session
comparison required `tool_schema_count`, every schema string, `tool_choice` and
`tool_choice_name` to match exactly, and the tool block lived in the head system
message (`append_tools_prompt`, `src/chat.c`). Pi adds and removes tool packages
(todo/goal/plan/MCP) and flips `tool_choice` between `auto` and `required`
during a conversation, so a turn that only changed tool metadata could not
continue: the whole 58-61K context was re-rendered and re-prefilled (~3.5
minutes on the ring).

## Decision

Tool metadata changes are rendered as a **trailing system tool-update block in
the continuation suffix**; the head system message is rendered once and stays
part of the cached prefix. Session compare treats the **message prefix** as the
only reset authority.

Rejected alternatives:

- **(a) Move the whole tool block trailing on every turn.** The suffix would
  repeat the full tool set each turn, growing the context by the tool set size
  per turn (a 60K budget with 6-8K of tools dies after a handful of turns) and
  changing the Qwen-official cold-start placement (tools belong in the system
  message). It also cannot keep ordinary turns byte-identical.
- **(c1) Runtime-level prefix surgery / per-tool-set KV forks.** The KV state
  behind the tool block depends on the block; keeping variants doubles state
  memory per tool set and requires `src/runtime.c` work for no format benefit.
- **(c2) Hash tool metadata into the head.** Detects the change but still has
  to express the new definitions somewhere; it reduces to (b) with extra state.

Option (b) wins because:

- Ordinary turns (metadata unchanged) render **byte-identical** to the previous
  behavior: no delta is emitted, no suffix growth.
- The delta is emitted only on a metadata change, and only differences are
  rendered: a `tool_choice` flip is one constraint line; adding a tool carries
  that tool's schema; removing a tool carries its name.
- The cold-start / fallback full render is untouched: the current tool set is
  still rendered as the Qwen ChatML system block at the head, so nothing in the
  non-continuation path moved or changed.
- Mid-conversation `system` messages are standard ChatML, so the delta block is
  in-distribution for the tokenizer and the model, and it sits before the new
  user/tool messages.

## Rendering rules (`fg_chat_render_tool_update`)

The block is wrapped in `<|im_start|>system ... <|im_end|>\n` and prepended to
the continuation suffix (which already ends with `<|im_start|>assistant`).
`previous` is the tool state the model last saw (the session transcript);
`current` is the request.

| Change | Block body |
|---|---|
| previous had no tools, current has tools | full `# Tools` declaration + Qwen format reminder + choice constraint (the head never carried the protocol) |
| tools removed to zero | `# Tools` + "No functions are available for this turn. Do not call any function; answer the user directly." |
| added / updated definitions | `# Tools` + "Tool configuration updated for this turn." + "The following functions are now available or have updated definitions:" + `<tools>` with only the changed schemas |
| removed tools | "The following functions are no longer available: name, name." |
| `tool_choice` changed | `Tool choice constraint: ...` with the same wording as the head; `auto` emits an explicit "Tool calls are optional this turn" release |
| metadata unchanged | no block at all (`*rendered = NULL`) |

Schema lists are compared order-insensitively; tool calls in historical
messages are unaffected because the continuation path never re-renders history.

## Session semantics (`api_public_session_prefix`)

Continuation requires the new request to extend the previous session's message
list with an exact prefix (role/content/tool_call_id/ids/names/arguments). The
tool metadata is no longer part of the reset decision: it is represented by the
delta, which is a pure function of the session transcript and the request. A
message edit, insertion, truncation or reorder still forces
`public-history-mismatch` + reset, so the delta cannot be used to smuggle a
tampered history past the cache.

If the continuation frontier is unavailable at the runtime (no pending EOS,
prefix miss), the request resets and falls back to the full render with the
current tool set in the head, so no stale delta survives a reset.

## Invariants

- Unchanged metadata across turns: byte-identical render and continuation
  output versus `0a76474` (corpus SHA-256 match, evidence
  `bc-250-dbg/results/tool-churn-20260917-1422/`).
- Text-only and vision paths untouched; media requests still reset as before.
- No new environment flags.

## System-message continuation deltas

Date: 2026-09-17. Branch `fix/system-delta-continuation` (base `001ad20`);
extends the rules above to the head system message. Invariants and the tool
renderer are unchanged.

### Problem

Pi enables/disables MCP servers by rewriting the head system prompt. A changed
`message[0].content` (or a role/position change at the head) failed
`api_public_session_prefix` and forced `public-history-mismatch` + full
re-prefill of the whole conversation.

### Rules

Only the **leading system run** is delta-eligible: the maximal prefix of
consecutive `system`/`developer` messages at the head of the transcript and of
the request. Everything after the two runs is compared exactly as before
(role/content/tool_call_id/tool call ids/names/arguments).

| Session (cached) | Request (new) | Result |
|---|---|---|
| run A | run B, same visible text | continue, no block (byte-identical) |
| run A | run B, different text | continue + system-delta block carrying B |
| no system run | run B | continue + system-delta block carrying B |
| run A | no system run (system removed / role moved away) | continue + removal block |
| run A | run B reordered, or different message count | continue + system-delta block carrying B (whole run replaced) |
| system/developer message after the first non-system message changed, added, removed or moved | — | reset (`message[i].role` / `message[i].content`) |
| request message count below transcript count (`messages=N->M`) | — | reset (history shrank; nothing to continue from) |

The run-length difference shifts the message alignment, so the continuation
slice starts at
`skip = transcript_count + (request_run - session_run)`; the delta block
represents the run itself. `messages=N->M` is checked first and keeps its
conservative reset. User, assistant and tool messages can never be edited,
inserted, removed or reordered without a reset.

### Rendering (`fg_chat_render_system_update`)

One `<|im_start|>system ... <|im_end|>\n` block is prepended to the
continuation suffix, before the tool-update block and the new messages:

| Change | Block body |
|---|---|
| replaced | `System instructions updated for this turn.\n\n` + new text |
| added (no prior system text) | new text (plain mid-conversation ChatML system block) |
| removed | `System instructions have been removed for this turn.` |
| unchanged visible text (`system`/`developer` alias or whitespace-only content) | no block at all |
| multiple leading messages | contents joined with `\n\n`; empty/whitespace-only contents skipped (the head drops them too) |

Reserved ChatML markers in the text are neutralized with a zero-width space,
same as every other rendered content. A system message that moved into the
new suffix (e.g. Pi appends it after the last user turn) is rendered by the
continuation itself; the delta only announces the head-run change.

### Invariants

- Unchanged system message across turns: byte-identical API transcript,
  headers and continuation versus `001ad20` (render + API corpora SHA-256
  match, evidence `bc-250-dbg/results/sys-delta-20260917-1655/`).
- Text-only and vision paths untouched; media requests still reset as before.
- No new environment flags.

## Validation

Local (WSL Ubuntu 24.04, gcc 13.3):

```bash
make tests/test_chat tests/test_api tests/test_prefix tests/test_chat_runtime
./tests/test_chat && ./tests/test_api
```

Byte-identity corpus (baseline `001ad20` vs the fix, renderer + API transcript):

```bash
bash bc-250-dbg/results/sys-delta-20260917-1655/run-corpus-diff.sh
```

Fleet A/B (only after the orchestrator clears the ring, after the prefix-diag
agent's work). With the fix built and serving on rank 0:

```powershell
& 'D:\looking-glass-labs\bc-250-dbg\test-fg-openai-tools.ps1' -BaseUrl http://192.168.42.42:8080/v1
pwsh -NoProfile -File 'D:\looking-glass-labs\bc-250-dbg\fleet\prefix-continuation-turn.ps1' `
    -FillerRepeats 4000 -Turns 4
pwsh -NoProfile -File 'D:\looking-glass-labs\bc-250-dbg\results\tool-churn-20260917-1422\tool-churn-ab.ps1' `
    -FillerRepeats 5200
pwsh -NoProfile -File 'D:\looking-glass-labs\bc-250-dbg\results\sys-delta-20260917-1655\sys-delta-ab.ps1' `
    -FillerRepeats 5200
```

Expected for the sys-delta script: turn 2 (system line added/removed, same
history) reports `X-Flash-Gordon-Prefix-Cache: hit`,
`X-Flash-Gordon-Reset-Reason: none`, reused tokens close to the turn-1 prompt
size, a single `<|im_start|>system` delta block in the suffix, and a sane
answer; a tampered turn still reports `miss` + `public-history-mismatch`.

