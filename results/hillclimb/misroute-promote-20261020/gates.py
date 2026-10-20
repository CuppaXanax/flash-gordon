import base64
import concurrent.futures
import datetime
import json
import pathlib
import statistics
import struct
import sys
import threading
import time
import urllib.request
import zlib

ROOT = pathlib.Path(__file__).parent
URL = "http://192.168.42.42:8080/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next-Uncensored"
SHORT = "/no_think What is 7 plus 5? Reply with only the number."
LONG = "/no_think Write an uninterrupted, detailed 3000-word explanation of how a mechanical clock works. Explain every component and keep elaborating; do not conclude early."


def request(messages, limit, temperature=0, barrier=None):
    body = dict(model=MODEL, messages=messages, temperature=temperature, top_p=0.95,
                max_tokens=limit, stream=True)
    if barrier:
        barrier.wait(timeout=10)
    start = time.time()
    chunks = []
    content = ""
    reasoning = ""
    usage = None
    finish = None
    done = False
    result_id = None
    metrics = {}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event, event
            result_id = event.get("id", result_id)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    chunks.append(time.time())
                content += delta.get("content", "")
                reasoning += delta.get("reasoning_content", "")
                finish = choice.get("finish_reason") or finish
            usage = event.get("usage", usage)
            metrics = event.get("flash_gordon", metrics)
    assert done and finish in ("stop", "length") and usage, (done, finish, usage)
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return dict(id=result_id, message=message, usage=usage, finish=finish,
                started=start, ended=time.time(), first_token=chunks[0],
                last_token=chunks[-1], delta_events=len(chunks),
                tps=(usage["completion_tokens"]-1)/(chunks[-1]-chunks[0]),
                flash_gordon=metrics)


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2))


def baseline():
    output = {"captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    output["short"] = request([dict(role="user", content=SHORT)], 32)
    output["greedy"] = []
    for _ in range(3):
        r = request([dict(role="user", content=LONG)], 320)
        assert r["usage"]["completion_tokens"] >= 300, r
        output["greedy"].append(r)
        save("baseline.json", output)
        print("BASELINE", r["id"], r["usage"], round(r["tps"], 3), flush=True)
    print("BASELINE_SHORT", output["short"]["message"], flush=True)


def gates():
    before = json.loads((ROOT / "baseline.json").read_text())
    output = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    output["short"] = request([dict(role="user", content=SHORT)], 32)
    assert output["short"]["message"] == before["short"]["message"], "greedy short changed"
    output["greedy"] = []
    for index in range(3):
        r = request([dict(role="user", content=LONG)], 320)
        assert r["message"] == before["greedy"][index]["message"], "greedy long changed"
        assert r["usage"]["completion_tokens"] >= 300
        output["greedy"].append(r)
        save("gates.json", output)
    old = statistics.median(r["tps"] for r in before["greedy"])
    new = statistics.median(r["tps"] for r in output["greedy"])
    output["greedy_tps"] = dict(before=old, after=new, ratio=new/old)
    assert new >= old*0.90, output["greedy_tps"]
    print("GREEDY", output["greedy_tps"], flush=True)

    messages = [dict(role="user", content=LONG + " This is conversation SOLO-MISROUTE.")]
    output["single"] = []
    for index in range(3):
        if index:
            messages.append(dict(role="user", content="/no_think Continue in the same detailed style for another 3000 words. Do not finish early."))
        r = request(messages, 352, 0.7)
        assert r["usage"]["completion_tokens"] >= 300, r
        if index:
            assert r["flash_gordon"]["reused_tokens"] > 0, "single continuation missed prefix"
        output["single"].append(r)
        messages.append(r["message"])
        save("gates.json", output)
        print("SINGLE", index, r["usage"], r["flash_gordon"], flush=True)

    # Admission preserves a larger cached conversation against short unrelated
    # requests. Make both new transcripts larger than the preceding SOLO cache.
    clock_reference = "Reference notes: clocks contain gears, shafts, pivots, bearings and springs.\n"*240
    ship_reference = "Reference notes: ships contain masts, sails, ropes, decks, keels and rudders.\n"*240
    conversations = [[dict(role="user", content=clock_reference + LONG + " Conversation CONCURRENT-A-MISROUTE.")],
                     [dict(role="user", content=ship_reference + "/no_think Write an uninterrupted 3000-word explanation of how a sailing ship works, covering every component in extensive detail. Do not conclude early. Conversation CONCURRENT-B-MISROUTE.")]]
    output["concurrent"] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for round_index in range(2):
            barrier = threading.Barrier(2)
            if round_index:
                for messages in conversations:
                    messages.append(dict(role="user", content="/no_think Continue for another 3000 words, expanding with more concrete details. Do not conclude early."))
            futures = [executor.submit(request, messages, 256, 0.7, barrier)
                       for messages in conversations]
            results = [future.result() for future in futures]
            for messages, r in zip(conversations, results):
                assert r["usage"]["completion_tokens"] >= 200, r
                if round_index:
                    assert r["flash_gordon"]["reused_tokens"] > 0, "concurrent continuation missed prefix"
                messages.append(r["message"])
            overlap = min(r["last_token"] for r in results)-max(r["first_token"] for r in results)
            output["concurrent"].append(dict(results=results, decode_overlap_seconds=overlap))
            save("gates.json", output)
            assert overlap > 1.0, ("no overlapping decode streams", results)
            print("CONCURRENT", round_index, [r["usage"] for r in results], "overlap", overlap, flush=True)

    chunk = lambda t, d: struct.pack(">I", len(d))+t+d+struct.pack(">I", zlib.crc32(t+d))
    png = (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0))+
           chunk(b"IDAT", zlib.compress((b"\0"+bytes([255, 0, 0])*224)*224))+chunk(b"IEND", b""))
    content = [dict(type="text", text="The secret password is BANANA-SEVEN. Reply with only the password and the image color. /no_think"),
               dict(type="image_url", image_url=dict(url="data:image/png;base64,"+base64.b64encode(png).decode()))]
    output["vision"] = request([dict(role="user", content=content)], 128)
    answer = output["vision"]["message"]["content"]
    assert "BANANA-SEVEN" in answer and "red" in answer.lower(), answer
    output["success"] = True
    output["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save("gates.json", output)
    print("VISION", answer, "ALL_API_GATES_PASS", flush=True)


if __name__ == "__main__":
    baseline() if sys.argv[1] == "baseline" else gates()
