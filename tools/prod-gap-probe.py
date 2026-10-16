"""Small descriptive serving sanity/trace workload; never an inferential gate."""
import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path


def request(messages, limit):
    body = {
        "model": "Qwen3.8-Flash-Next-Uncensored", "messages": messages,
        "temperature": 0, "top_p": 1, "top_k": 1, "seed": 1729,
        "max_tokens": limit, "stream": False,
    }
    wire = json.dumps(body, separators=(",", ":")).encode()
    started = time.perf_counter()
    req = urllib.request.Request("http://192.168.42.42:8080/v1/chat/completions",
                                 data=wire, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        payload = json.load(response)
        headers = dict(response.headers)
    elapsed = time.perf_counter() - started
    text = payload["choices"][0]["message"]["content"]
    return {
        "wall_seconds": elapsed, "usage": payload["usage"], "headers": headers,
        "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "request_sha256": hashlib.sha256(wire).hexdigest(),
        "client_wall_tps": payload["usage"]["completion_tokens"] / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--salt", default=None)
    parser.add_argument("--prime-only", action="store_true")
    parser.add_argument("--decode-from", type=Path)
    args = parser.parse_args()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.decode_from:
        source = json.loads(args.decode_from.read_text())
        decode = request(source["rows"][-1]["follow"], 128)
        args.output.write_text(json.dumps({
            "inferential": False, "prime_source": str(args.decode_from), "decode": decode,
        }, indent=2) + "\n")
        print("actual_tokens", decode["usage"]["completion_tokens"],
              "engine_tps", decode["headers"].get("X-Flash-Gordon-Decode-TPS"), flush=True)
        return
    for repeat in range(args.repeats):
        for context in args.contexts:
            salt = args.salt or uuid.uuid4().hex
            nonce = hashlib.sha256(f"{salt}/{repeat}/{context}".encode()).hexdigest()[:32]
            messages = [
                {"role": "system", "content": f"Isolated benchmark nonce {nonce}. Answer directly."},
                {"role": "user", "content": f"/no_think nonce {nonce}. "
                 + " hello" * max(0, context - 64) + "\nOutput exactly Ready."},
            ]
            cold = request(messages, 1)
            follow = messages + [
                {"role": "assistant", "content": cold["text"]},
                {"role": "user", "content": "/no_think Write a long numbered list of distinct practical software tasks. Continue until the token limit."},
            ]
            if args.prime_only:
                args.output.write_text(json.dumps({
                    "inferential": False, "rows": [{"cold": cold, "follow": follow,
                     "context": context, "salt": salt}],
                }, indent=2) + "\n")
                return
            decode = request(follow, 128)
            rows.append({"repeat": repeat, "context": context, "salt": salt,
                         "cold": cold, "decode": decode})
            args.output.write_text(json.dumps({
                "inferential": False, "started_workload": "cold then cached follow-on",
                "rows": rows,
            }, indent=2) + "\n")
            print(repeat, context, "actual_tokens", decode["usage"]["completion_tokens"],
                  "engine_tps", decode["headers"].get("X-Flash-Gordon-Decode-TPS"),
                  "wall_tps", round(decode["client_wall_tps"], 3), flush=True)


if __name__ == "__main__":
    main()
