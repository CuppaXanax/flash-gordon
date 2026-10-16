"""Curated real-text corpus and strict acceptance/greedy receipt analysis."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

CHAT = [
    "Plan a week in Kyoto for a first-time visitor who likes quiet gardens, bookstores, and public transit.",
    "Explain how a city library can organize a welcoming monthly book club for adults with different reading interests.",
    "Help me compare renting and buying an apartment using a worked hypothetical monthly budget.",
    "Teach a curious teenager how seasons, lunar phases, and eclipses differ, with everyday analogies.",
    "Design a beginner balcony vegetable garden with limited sunlight and a small budget.",
    "Explain the water cycle and how a city can reduce stormwater runoff through neighborhood design.",
    "Create a detailed study plan for an adult learning conversational Spanish over twelve weeks.",
    "Suggest a low-stress system for organizing family photographs, including filenames, backups, and printed albums.",
    "Explain the history of public libraries and how digital access changes their role.",
    "Help a remote team improve asynchronous communication across time zones with concrete examples.",
    "Create a practical moving checklist for someone relocating to a small apartment in another city.",
    "Compare trains, buses, and bicycles for an everyday urban commute, considering cost and reliability.",
    "Describe how to host a community repair cafe, including volunteer roles and an accessible registration process.",
    "Teach the difference between correlation and causation using several realistic everyday examples.",
    "Develop a welcoming orientation guide for volunteers at a neighborhood food pantry.",
    "Explain how orchestral instruments create different timbres and suggest a listening guide for a newcomer.",
    "Help me design a weekly meal-planning routine for a vegetarian household with varied tastes.",
    "Compare three ways a small museum could make its exhibitions more accessible to visitors.",
    "Write an engaging explanation of plate tectonics, mountain formation, and geological time.",
    "Plan a weekend family reunion with indoor activities, dietary options, and a modest budget.",
    "Explain how to evaluate an online source's credibility without relying on its visual design.",
    "Design a six-month reading challenge that mixes fiction, history, science, and poetry.",
    "Describe the process of producing a community newsletter from gathering stories to distribution.",
    "Help a first-time manager prepare thoughtful one-on-one conversations with a new team.",
]
CODE = [
    "Python: implement an LRU cache with a dictionary and doubly linked list, type annotations, and unittest coverage.",
    "C17: implement a growable UTF-8 byte buffer with overflow checks, explicit ownership, and tests for allocation failures.",
    "Go: implement a bounded worker pool with context cancellation, orderly shutdown, and race-safe unit tests.",
    "Rust: implement a command-line CSV summarizer with structured errors, streaming input, and unit tests.",
    "TypeScript: implement a debounced search controller that cancels stale requests, with fake-clock tests.",
    "Java: implement an immutable money value type with currency checking, rounding rules, and JUnit tests.",
    "C++20: implement a move-only scope guard and demonstrate exception safety with tests.",
    "Python: implement stable topological sorting with cycle reporting and tests for disconnected graphs.",
    "Go: implement a token-bucket rate limiter with injectable time and deterministic tests.",
    "SQL and Python: design a SQLite migration runner with transactions, checksums, and restart-safe tests.",
    "C17: implement a ring buffer for single-threaded byte streams, including wraparound and full-buffer tests.",
    "JavaScript: implement an event emitter with once, unsubscribe, and safe mutation during dispatch, plus tests.",
    "Rust: implement a generic binary heap with push, pop, peek, and property-style test cases.",
    "Python: implement a streaming JSON Lines validator that reports line-specific errors without loading the file.",
    "TypeScript: implement a dependency-free finite-state machine with typed events and transition tests.",
    "Java: implement a trie with prefix completion, deterministic ordering, and Unicode-aware tests.",
    "Go: implement a streaming file checksum utility with cancellation, progress reporting, and tests.",
    "C++20: implement a small arena allocator with alignment checks and tests for exhaustion.",
    "Python: implement a retry helper with exponential backoff, jitter injection, and deterministic unit tests.",
    "C17: implement a command-line argument parser supporting long options, validation, and clear errors, with tests.",
    "Rust: implement a line-oriented configuration parser with comments, duplicate-key errors, and tests.",
    "TypeScript: implement a bounded async queue with backpressure and graceful close, including concurrency tests.",
    "Python: implement a filesystem-independent directory-tree comparison using injected readers and thorough tests.",
    "Go: implement an in-memory TTL cache with an injectable clock and explicit cleanup, including concurrent tests.",
]


def corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(24):
        for category, topics in (("chat", CHAT), ("code", CODE)):
            case = f"{category}-{i:02d}"
            instruction = (
                "Give a detailed, useful answer of at least 900 words, with concrete examples and clear sections."
                if category == "chat" else
                "Provide a complete implementation followed by extensive runnable tests and explanation. "
                "Include at least 100 lines of code; do not replace important parts with ellipses."
            )
            text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{topics[i]}\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            (directory / f"{case}.txt").write_bytes(text.encode("utf-8"))
            records.append(dict(id=case, category=category, sha256=hashlib.sha256(text.encode()).hexdigest()))
    (directory / "cases.tsv").write_text("".join(
        f"{r['id']}\t{r['category']}\t/home/xander/hc-mtp/corpus/{r['id']}.txt\n"
        for r in records), encoding="utf-8")
    (directory / "corpus.json").write_text(json.dumps(dict(
        origin="Original curated realistic chat/code requests; not production user traffic",
        template="production think-off chat template", generated_tokens=259,
        acceptance_anchors=256, minimum_per_category=20, reserves_per_category=4,
        cases=records), indent=2) + "\n", encoding="utf-8")


def summarize(rows, greedy=None):
    groups = {}
    for row in rows:
        if row["category"] not in ("chat", "code") or not 0 <= row["position"] < 256:
            raise ValueError("invalid category/position")
        records = groups.setdefault(row["case"], {})
        if row["position"] in records:
            raise ValueError("duplicate position")
        matches = [int(a == b) for a, b in zip(row["draft"], row["target"], strict=True)]
        prefix, live = [], 1
        for match in matches:
            live &= match
            prefix.append(live)
        if len(matches) != 3 or row["match"] != matches or row["prefix"] != prefix:
            raise ValueError("inconsistent draft acceptance")
        records[row["position"]] = row
    counts = Counter()
    for records in groups.values():
        if set(records) != set(range(256)):
            raise ValueError("incomplete per-position capture")
        categories = {r["category"] for r in records.values()}
        if len(categories) != 1:
            raise ValueError("category changed within case")
        counts[next(iter(categories))] += 1
    if counts != Counter(chat=20, code=20):
        raise ValueError("need exactly 20 eligible cases per category")
    result = dict(measured=True, sampling="greedy", generated_tokens_per_case=259,
                  anchors_per_case=256, cases=dict(counts), serving_mtp_linked=False,
                  acceptance={}, greedy=None)
    for category in ("chat", "code", "all"):
        selected = [rs for rs in groups.values() if category == "all" or rs[0]["category"] == category]
        flat = [r for rs in selected for r in rs.values()]
        output = {}
        for k in (1, 2, 3):
            positions = []
            for j in range(k):
                attempts = sum(1 if j == 0 else r["prefix"][j - 1] for r in flat)
                accepted = sum(r["prefix"][j] for r in flat)
                positions.append(dict(draft_position=j + 1, attempts=attempts, accepted=accepted,
                                      conditional_acceptance=accepted / attempts if attempts else None,
                                      prefix_acceptance=accepted / len(flat)))
            passes = committed = 0
            for rs in selected:
                p = 0
                while p < 255:
                    step = min(1 + sum(rs[p]["prefix"][:k]), 255 - p)
                    p += step
                    committed += step
                    passes += 1
            output[f"k{k}"] = dict(draft_positions=positions,
                anchored_committed_tokens_per_pass=1 + sum(r["prefix"][:k][j]
                    for r in flat for j in range(k)) / len(flat),
                fixed_k_policy_walk=dict(passes=passes, committed_tokens=committed,
                                         tokens_per_pass=committed / passes),
                generation_positions=[dict(position=p, samples=len(selected),
                    prefix_acceptance=[sum(rs[p]["prefix"][j] for rs in selected) / len(selected)
                                       for j in range(k)]) for p in range(256)])
        result["acceptance"][category] = output
    if greedy is not None:
        expected = {(case, k) for category in ("chat", "code")
                    for case in [c for c, rs in groups.items() if rs[0]["category"] == category][:4]
                    for k in (1, 2, 3)}
        actual = [(r["case"], r["k"]) for r in greedy]
        if len(actual) != 24 or set(actual) != expected:
            raise ValueError("greedy gate coverage must be four chat/four code at each k")
        if any(not r["equal"] or r["tokens"] != 256 or r["rollbacks"] < 1 or
               r["verifier"] != "causal-N1-reference" for r in greedy):
            raise ValueError("greedy/rollback gate failed")
        result["greedy"] = dict(passed=True, cases_per_category=4, k=[1, 2, 3],
            paired_runs=24, tokens_per_run=256, forced_rejection_each_run=True,
            verifier="causal N=1 ring reference; not small-N serving-path qualification",
            receipts=greedy)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("capture", type=Path)
    p.add_argument("--require-greedy", action="store_true")
    args = parser.parse_args()
    if args.command == "corpus":
        corpus(args.directory)
        return
    rows = [json.loads(s) for s in (args.capture / "acceptance.jsonl").read_text().splitlines()]
    greedy = None
    if args.require_greedy:
        greedy = [json.loads(s) for s in (args.capture / "greedy.jsonl").read_text().splitlines()]
    result = summarize(rows, greedy)
    (args.capture / "acceptance-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
