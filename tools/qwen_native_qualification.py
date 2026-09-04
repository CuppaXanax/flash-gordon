#!/usr/bin/env python3
"""Bounded, append-only Qwen native-context collection over the public API.

This is a collector, not a benchmark shortcut: every prompt is a prefix of the
previous prompt and every user chunk is a unique contiguous slice of --corpus.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "fg.qwen.qual.client.v1"
DEFAULT_CHECKPOINTS = "4096,8192,16384,32768,65536,131072,196608,261888"


def _json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _request(base_url, body, timeout, ordinal):
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    started = time.monotonic()
    first_event = None
    chunks = []
    usage = None
    request_id = "client-%d" % ordinal
    finish = None
    status = 0
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            pending = b""
            while True:
                part = response.read(4096)
                if not part:
                    break
                pending += part
                lines = pending.split(b"\n")
                pending = lines.pop()
                for raw in lines:
                    line = raw.rstrip(b"\r")
                    if not line.startswith(b"data: "):
                        continue
                    text = line[6:].decode("utf-8", "replace")
                    if text == "[DONE]":
                        continue
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError as exc:
                        error = "invalid SSE JSON: %s" % exc
                        continue
                    request_id = str(event.get("id", request_id))
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    has_generated_output = any(
                        isinstance(delta.get(field), str) and bool(delta.get(field))
                        for field in ("content", "reasoning_content")
                    ) or bool(delta.get("tool_calls"))
                    if first_event is None and has_generated_output:
                        first_event = time.monotonic()
                    if isinstance(delta.get("content"), str):
                        chunks.append(delta["content"])
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
            if pending.startswith(b"data: "):
                try:
                    event = json.loads(pending[6:].decode("utf-8", "replace"))
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                except (ValueError, UnicodeDecodeError):
                    pass
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = "HTTP %d" % exc.code
        try:
            error += ": " + exc.read().decode("utf-8", "replace")[:512]
        except Exception:
            pass
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)
    wall_ms = (time.monotonic() - started) * 1000.0
    return {
        "request_id": request_id,
        "http_status": status,
        "finish_reason": finish,
        "text": "".join(chunks),
        "usage": usage,
        "ttft_ms": None if first_event is None else (first_event - started) * 1000.0,
        "total_wall_ms": wall_ms,
        "error": error,
    }


def _parse_checkpoints(value):
    points = sorted(set(int(item) for item in value.split(",") if item.strip()))
    if not points or any(point <= 0 for point in points):
        raise ValueError("checkpoints must be positive comma-separated token counts")
    return points


def collect(args):
    corpus = Path(args.corpus).read_text(encoding="utf-8")
    checkpoints = _parse_checkpoints(args.checkpoints)
    terminal_context = args.native_context - args.headroom
    if terminal_context <= 0 or checkpoints[-1] > terminal_context:
        raise ValueError("checkpoints must stay below native-context minus headroom")
    if args.repeats < 1 or args.chunk_chars < 1 or args.max_tokens < 1:
        raise ValueError("repeats, chunk-chars, and max-tokens must be positive")
    deadline = time.monotonic() + args.wall_budget_minutes * 60.0
    collection_started = time.monotonic()
    messages = [{"role": "system", "content":
                 "You are a coding assistant. Reply briefly and do not use tools."}]
    corpus_offset = 0
    last_context = 0
    ordinal = 0
    records = []
    failures = []
    corpus_ranges = set()
    next_checkpoint = 0
    chunk_chars = args.chunk_chars

    with Path(args.output).open("w", encoding="utf-8") as output:
        while next_checkpoint < len(checkpoints):
            target = checkpoints[next_checkpoint]
            if time.monotonic() >= deadline:
                failures.append("wall budget exhausted")
                break
            if corpus_offset >= len(corpus):
                failures.append("corpus exhausted before checkpoint %d" % target)
                break
            end = min(len(corpus), corpus_offset + chunk_chars)
            start = corpus_offset
            messages.append({"role": "user", "content": corpus[start:end]})
            body = {"model": args.model, "messages": messages, "stream": True,
                    "temperature": args.temperature, "top_p": args.top_p,
                    "top_k": args.top_k, "max_tokens": args.max_tokens}
            ordinal += 1
            result = _request(args.base_url, body,
                              max(1.0, min(args.request_timeout_seconds,
                                           deadline - time.monotonic())), ordinal)
            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            record = {"schema": SCHEMA, "request_id": result["request_id"],
                      "checkpoint_target": target, "prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens, "total_tokens": total_tokens,
                      "client_ttft_ms": result["ttft_ms"],
                      "client_total_wall_ms": result["total_wall_ms"],
                      "http_status": result["http_status"],
                      "finish_reason": result["finish_reason"],
                      "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                   "top_k": args.top_k},
                      "corpus_char_start": start, "corpus_char_end": end,
                      "checkpoint_crossed": bool(prompt_tokens is not None and
                                                 prompt_tokens >= target),
                      "partial_output": result["text"]}
            if (start, end) in corpus_ranges:
                failures.append("duplicate corpus range %d:%d" % (start, end))
            elif end > start:
                corpus_ranges.add((start, end))
            if result["error"]:
                record["error"] = result["error"]
            elapsed = time.monotonic() - collection_started
            if isinstance(prompt_tokens, int):
                observed_rate = prompt_tokens / elapsed if elapsed > 0.0 else None
                estimated_remaining = ((terminal_context - prompt_tokens) / observed_rate
                                       if observed_rate and prompt_tokens < terminal_context else 0.0)
                record["progress_elapsed_seconds"] = elapsed
                record["observed_session_tokens_per_second"] = observed_rate
                record["estimated_remaining_seconds"] = estimated_remaining
            output.write(_json_line(record) + "\n")
            output.flush()
            records.append(record)
            if failures:
                break
            if result["error"] or result["http_status"] != 200 or not result["finish_reason"]:
                failures.append("request %s failed" % record["request_id"])
                break
            if prompt_tokens is None or total_tokens is None or completion_tokens is None:
                failures.append("request %s did not report usage" % record["request_id"])
                break
            if prompt_tokens < last_context or total_tokens < prompt_tokens:
                failures.append("non-monotonic context/accounting at %s" % record["request_id"])
                break
            previous_context = last_context
            print("progress: context=%d target=%d elapsed=%.1fs estimated_remaining=%.1fs "
                  "(observed end-to-end rate)" %
                  (prompt_tokens, target, elapsed, estimated_remaining), file=sys.stderr)
            corpus_offset = end
            last_context = prompt_tokens
            messages.append({"role": "assistant", "content": result["text"]})
            if prompt_tokens >= target:
                for _ in range(args.repeats - 1):
                    if time.monotonic() >= deadline:
                        failures.append("wall budget exhausted")
                        break
                    repeat_body = {"model": args.model, "messages": messages, "stream": True,
                                   "temperature": args.temperature, "top_p": args.top_p,
                                   "top_k": args.top_k, "max_tokens": args.max_tokens}
                    ordinal += 1
                    repeat = _request(args.base_url, repeat_body,
                                      max(1.0, min(args.request_timeout_seconds,
                                                   deadline - time.monotonic())), ordinal)
                    ru = repeat.get("usage") or {}
                    repeat_record = {"schema": SCHEMA, "request_id": repeat["request_id"],
                                     "checkpoint_target": target,
                                     "prompt_tokens": ru.get("prompt_tokens"),
                                     "completion_tokens": ru.get("completion_tokens"),
                                     "total_tokens": ru.get("total_tokens"),
                                     "client_ttft_ms": repeat["ttft_ms"],
                                     "client_total_wall_ms": repeat["total_wall_ms"],
                                     "http_status": repeat["http_status"],
                                     "finish_reason": repeat["finish_reason"],
                                     "sampling": {"temperature": args.temperature,
                                                  "top_p": args.top_p, "top_k": args.top_k},
                                     "corpus_char_start": corpus_offset,
                                     "corpus_char_end": corpus_offset,
                                     "checkpoint_crossed": True,
                                     "partial_output": repeat["text"], "prefix_continuation": True}
                    if repeat["error"]:
                        repeat_record["error"] = repeat["error"]
                    output.write(_json_line(repeat_record) + "\n")
                    output.flush()
                    records.append(repeat_record)
                    if repeat["error"] or repeat["http_status"] != 200:
                        failures.append("continuation %s failed" % repeat_record["request_id"])
                        break
                    if (repeat_record["prompt_tokens"] is None or
                            repeat_record["total_tokens"] is None or
                            repeat_record["prompt_tokens"] < last_context or
                            repeat_record["total_tokens"] < repeat_record["prompt_tokens"]):
                        failures.append("non-monotonic continuation accounting at %s" %
                                        repeat_record["request_id"])
                        break
                    last_context = repeat_record["prompt_tokens"]
                    messages.append({"role": "assistant", "content": repeat["text"]})
                next_checkpoint += 1
            else:
                remaining = max(1, target - prompt_tokens)
                observed = max(1.0, (end - start) / max(1, prompt_tokens - previous_context))
                # The estimate is only a chunk-size hint; actual usage is authoritative.
                chunk_chars = max(256, min(args.max_chunk_chars,
                                           int(remaining * observed * args.chunk_safety)))
        if next_checkpoint < len(checkpoints) and not failures:
            failures.append("checkpoint collection stopped before final target")
    if failures:
        print("qualification collection failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wall-budget-minutes", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0,
                        help="maximum silence allowed for one HTTP request")
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--chunk-chars", type=int, default=8192)
    parser.add_argument("--max-chunk-chars", type=int, default=65536)
    parser.add_argument("--chunk-safety", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--native-context", type=int, default=262144)
    parser.add_argument("--headroom", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if (args.wall_budget_minutes <= 0 or args.request_timeout_seconds <= 0 or
                args.top_k < 1 or not 0 < args.top_p <= 1):
            raise ValueError("invalid budget or sampling controls")
        return collect(args)
    except (OSError, ValueError) as exc:
        print("qualification collection failed: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
