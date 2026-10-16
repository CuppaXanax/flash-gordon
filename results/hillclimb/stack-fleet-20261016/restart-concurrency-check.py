import concurrent.futures
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

base = "http://192.168.42.42:8080"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
barrier = threading.Barrier(2)
nonce = uuid.uuid4().hex
start = time.perf_counter()


def run(index):
    body = {"model": "Qwen3.8-Flash-Next-Uncensored", "temperature": 0,
            "max_tokens": 96, "stream": False, "user": f"restart-stream-{index}",
            "messages": [{"role": "user", "content":
                          f"/no_think Verification nonce {nonce}, stream {index}. "
                          "Write a long numbered list of distinct practical software tasks. "
                          "Continue until the token limit."}]}
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    barrier.wait()
    sent = time.perf_counter()
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
        headers = dict(response.headers)
    finished = time.perf_counter()
    tokens = result["usage"]["completion_tokens"]
    text = result["choices"][0]["message"]["content"]
    assert tokens >= 32 and text.strip(), result
    return {"stream": index, "completion_tokens": tokens,
            "start_seconds": sent - start, "finish_seconds": finished - start,
            "wall_seconds": finished - sent, "headers": headers,
            "text_excerpt": text[:160]}


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(run, range(2)))
health = json.load(opener.open(base + "/health", timeout=10))
models = json.load(opener.open(base + "/v1/models", timeout=10))
assert health["status"] == "ok"
assert models["data"][0]["id"] == "Qwen3.8-Flash-Next-Uncensored"
report = {"descriptive_restart_verification_not_gate_data": True,
          "streams": rows, "health": health, "models": models,
          "actual_aggregate_tps": sum(r["completion_tokens"] for r in rows) /
                                  max(r["finish_seconds"] for r in rows)}
(Path(".build") / "stack-live" / "final-concurrency.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"streams": len(rows), "actual_tokens": [r["completion_tokens"] for r in rows],
                  "actual_aggregate_tps": report["actual_aggregate_tps"], "health": health}))
