#!/usr/bin/env python3
"""Compare full-vocabulary teacher-forced logit dumps; no top-k approximation."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

PROMPTS = [
    "Explain why the sky appears blue, briefly.",
    "Write a friendly reply declining a meeting invitation.",
    "Summarize the tradeoffs between accuracy and latency.",
    "A user is upset about a late order. Draft a helpful response.",
    "What assumptions are needed to prove a binary search correct?",
    "Write a Python function that merges two sorted lists.",
    "Find the bug: for (size_t i = n - 1; i >= 0; --i) use(a[i]);",
    "Implement an SQL query for the second highest salary.",
    "Explain Rust ownership with a small example.",
    "Write a TypeScript discriminated union for an HTTP result.",
    "Translate to French: The train arrives tomorrow morning.",
    "用中文解释什么是递归，并给出一个简单例子。",
    "日本語で二分探索の仕組みを説明してください。",
    "Explica en español la diferencia entre ser y estar.",
    "اشرح باختصار لماذا نستخدم الاختبارات الآلية.",
    "Объясни разницу между процессом и потоком.",
    "Solve: a train travels 180 km in 2.5 hours. What is its average speed?",
    "Explain the difference between a correlation and a causal effect.",
    "Describe how you would test image upload validation without an image.",
    "A document contains text in several languages. Design a Unicode-safe index.",
]


def contexts():
    # Fixed textual prefixes ensure neither model can feed its own predictions back.
    for i, prompt in enumerate(PROMPTS):
        for j, suffix in enumerate(("", "First", "First, consider", "First, consider the")):
            yield f"{i:02}-{j}", (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{suffix}")


def read(path):
    raw = path.read_bytes()
    if raw[:8] != b"FGLOG01\n":
        raise ValueError(f"bad logit header: {path}")
    vocab, = struct.unpack("<I", raw[8:12])
    if len(raw) != 12 + vocab * 4 or vocab != 248320:
        raise ValueError(f"not a full Qwen vocabulary: {path}")
    values = array.array("f")
    values.frombytes(raw[12:])
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"non-finite logits: {path}")
    return values


def compare(teacher, student):
    mt, ms = max(teacher), max(student)
    zt = math.fsum(math.exp(x-mt) for x in teacher)
    zs = math.fsum(math.exp(x-ms) for x in student)
    logzt, logzs = mt+math.log(zt), ms+math.log(zs)
    kld = math.fsum(math.exp(t-logzt)*((t-logzt)-(s-logzs))
                    for t, s in zip(teacher, student))
    top_t = max(range(len(teacher)), key=teacher.__getitem__)
    top_s = max(range(len(student)), key=student.__getitem__)
    return max(0.0, kld), top_t == top_s


def validate_capture_corpus(directory, items):
    recorded=json.loads((directory/"corpus.json").read_text(encoding="utf-8-sig"))
    if (not isinstance(recorded,list) or any(not isinstance(x,list) or len(x)!=2
            or not all(isinstance(value,str) for value in x) for x in recorded)):
        raise ValueError("invalid capture corpus metadata")
    by_id=dict(recorded)
    if len(by_id)!=len(recorded) or any(by_id.get(key)!=text for key,text in items):
        raise ValueError("capture corpus does not match frozen text")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contexts", type=Path, help="write fixed teacher-forced contexts as JSONL")
    p.add_argument("--corpus", type=Path, help="frozen JSON [id, rendered-context] pairs")
    p.add_argument("--framed-contexts", type=Path, help="write length-framed UTF-8 contexts for resident batch capture")
    p.add_argument("--import-numbered", type=Path, help="validate and rename a resident batch's numbered dumps")
    p.add_argument("--teacher", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--capture", type=Path, help="capture directory, using --command")
    p.add_argument("--command", help='JSON argv; substitute {prompt} and {output}; must run eval --generate 0 and retrieve worker dump')
    args = p.parse_args()
    items = json.loads(args.corpus.read_text(encoding="utf-8-sig")) if args.corpus else list(contexts())
    if (not items or len(items)>1024 or any(not isinstance(x,(list,tuple)) or len(x)!=2
        or not all(isinstance(y,str) and y for y in x) or "\0" in x[1] for x in items)
        or len({x[0] for x in items}) != len(items)
        or any(not all(c.isalnum() or c in "-_" for c in x[0]) for x in items)):
        raise ValueError("invalid or duplicate frozen corpus")
    if args.framed_contexts:
        with args.framed_contexts.open("xb") as file:
            file.write(b"FGEVAL01")
            for _, prompt in items:
                raw = prompt.encode("utf-8")
                file.write(struct.pack("<I", len(raw)))
                file.write(raw)
        return
    if args.import_numbered:
        root = args.import_numbered
        numbered = sorted(root.glob("[0-9][0-9][0-9][0-9].fgl"))
        if len(numbered) != len(items):
            raise ValueError(f"expected {len(items)} fresh full-vocabulary dumps")
        for i, (key, _) in enumerate(items):
            source = root/f"{i:04}.fgl"
            read(source)
            if root.joinpath(key+".fgl").exists():
                raise ValueError(f"refuse stale imported capture: {key}")
        for i, (key, _) in enumerate(items):
            root.joinpath(f"{i:04}.fgl").rename(root/(key+".fgl"))
        root.joinpath("corpus.json").write_text(json.dumps(items, ensure_ascii=False)+"\n", encoding="utf-8")
        return
    if args.capture:
        if not args.command:
            p.error("--capture requires --command")
        command = json.loads(args.command)
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            p.error("--command must be a JSON array of strings")
        args.capture.mkdir(parents=True, exist_ok=True)
        for key, prompt in items:
            output = (args.capture/(key+".fgl")).resolve()
            if output.exists():
                raise ValueError(f"refuse stale capture: {output}")
            argv = [x.replace("{prompt}", prompt).replace("{output}", str(output))
                    for x in command]
            subprocess.run(argv, check=True)
            read(output)
        args.capture.joinpath("corpus.json").write_text(
            json.dumps(items, ensure_ascii=False)+"\n", encoding="utf-8")
        return
    if args.contexts:
        args.contexts.write_text("".join(json.dumps(dict(id=i, prompt=t), ensure_ascii=False)
                                       + "\n" for i, t in items), encoding="utf-8")
        return
    if not all((args.teacher, args.candidate, args.output)):
        p.error("--teacher, --candidate and --output are required")
    for directory in (args.teacher,args.candidate):
        validate_capture_corpus(directory,items)
    corpus = hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest()
    rows = []
    for key, _ in items:
        kld, top1 = compare(read(args.teacher/(key+".fgl")),
                            read(args.candidate/(key+".fgl")))
        rows.append(dict(id=key, kld=kld, top1=top1))
    losses = sorted(r["kld"] for r in rows)
    mean, p99 = math.fsum(losses)/len(losses), losses[math.ceil(.99*len(losses))-1]
    agreement = sum(r["top1"] for r in rows)/len(rows)
    passed = mean <= .01 and p99 <= .05 and agreement >= .97
    report = dict(corpus_sha256=corpus, samples=len(rows), full_vocab=248320,
                  mean_kld=mean, p99_kld=p99, top1_agreement=agreement,
                  passed=passed, rows=rows)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
