#!/usr/bin/env python3
"""Non-MLA LMCache CacheGen serde end-to-end client.

Runs inside the vLLM container. Two phases, driven by a host orchestrator that
restarts the engine between them so that L1 (CPU) cache is empty on retrieval:

  populate : send a long, chunk-unique prompt to fill L1 + the remote backend,
             wait for the remote directory size to settle, report remote bytes.
  retrieve : send the *same* prompt against a freshly started engine (L1 empty),
             measure TTFT / generation throughput / cached tokens, dump output.

Only stdlib is used so it runs on any vLLM image.
"""

import argparse
import json
import os
import time
import urllib.request

# ---------------------------------------------------------------- HTTP helpers


def _post(base_url, path, payload, timeout=3600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def wait_health(base_url, timeout=900):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(3)
    raise RuntimeError(f"health check timed out after {timeout}s (last={last})")


def tokenize(base_url, model, text):
    with _post(base_url, "/tokenize", {"model": model, "prompt": text}, timeout=120) as r:
        return json.loads(r.read())["count"]


# ---------------------------------------------------------------- prompt build


def build_prompt(base_url, model, target_tokens):
    """Build a deterministic prompt whose 256-token chunks are all distinct.

    Distinct chunks matter: identical chunks would dedup into one cache entry
    and distort both the remote-byte measurement and the hit count.
    """
    lines = [
        f"Fact {i}: record {i}, code {i * 7919 % 9973}, group {i % 53}, "
        f"weight {i * 104729 % 8191}, label L{i * 31 % 4099}."
        for i in range(2000)
    ]
    text = "\n".join(lines)
    total = tokenize(base_url, model, text)
    per_line = total / len(lines)
    need = max(1, int(target_tokens / per_line))
    text = "\n".join(lines[:need])
    # one correction pass to land at/above the target
    tokens = tokenize(base_url, model, text)
    while tokens < target_tokens and need < len(lines):
        need += max(1, int((target_tokens - tokens) / per_line))
        text = "\n".join(lines[:need])
        tokens = tokenize(base_url, model, text)
    return text, tokens


def send_completion(base_url, model, prompt, max_tokens=32, timeout=3600):
    """Streaming completion; returns TTFT and aggregate metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    first = None
    text_parts = []
    last_chunk = None
    with _post(base_url, "/v1/completions", payload, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if obj.get("choices"):
                piece = obj["choices"][0].get("text")
                if piece:
                    if first is None:
                        first = time.time() - t0
                    text_parts.append(piece)
            if obj.get("usage"):
                last_chunk = obj
    wall = time.time() - t0
    usage = (last_chunk or {}).get("usage", {}) or {}
    details = usage.get("prompt_tokens_details", {}) or {}
    return {
        "ttft_s": first,
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "output_text": "".join(text_parts),
    }


# ---------------------------------------------------------------- remote bytes


def remote_bytes(path):
    total = 0
    files = 0
    for root, _dirs, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
                files += 1
            except OSError:
                pass
    return total, files


def wait_remote_settle(path, quiet_s=8.0, timeout=300.0):
    t0 = time.time()
    last = -1
    stable_since = None
    while time.time() - t0 < timeout:
        cur, files = remote_bytes(path)
        if cur != last:
            last = cur
            stable_since = time.time()
        elif stable_since is not None and time.time() - stable_since >= quiet_s:
            return cur, files
        time.sleep(2)
    return last, remote_bytes(path)[1]


# ---------------------------------------------------------------- phases


def phase_populate(args):
    health_s = wait_health(args.base_url)
    prompt, tokens = build_prompt(args.base_url, args.model, args.prompt_tokens)
    pop = send_completion(args.base_url, args.model, prompt, max_tokens=args.max_tokens)
    size, files = wait_remote_settle(args.remote_dir)
    result = {
        "phase": "populate",
        "health_wait_s": round(health_s, 2),
        "prompt_tokens": tokens,
        "remote_dir": args.remote_dir,
        "remote_bytes": size,
        "remote_files": files,
        "populate_ttft_s": pop["ttft_s"],
        "populate_wall_s": round(pop["wall_s"], 3),
        "populate_cached_tokens": pop["cached_tokens"],
        "output_text": pop["output_text"],
    }
    emit(args, result)


def phase_retrieve(args):
    health_s = wait_health(args.base_url)
    prompt, tokens = build_prompt(args.base_url, args.model, args.prompt_tokens)
    reps = []
    for i in range(args.reps):
        r = send_completion(args.base_url, args.model, prompt, max_tokens=args.max_tokens)
        reps.append(
            {
                "rep": i,
                "ttft_s": r["ttft_s"],
                "wall_s": round(r["wall_s"], 3),
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "output_text": r["output_text"],
            }
        )
    size, files = remote_bytes(args.remote_dir)
    result = {
        "phase": "retrieve",
        "health_wait_s": round(health_s, 2),
        "prompt_tokens": tokens,
        "remote_dir": args.remote_dir,
        "remote_bytes": size,
        "remote_files": files,
        "reps": reps,
    }
    emit(args, result)


def emit(args, result):
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["populate", "retrieve"])
    ap.add_argument("--base-url", default="http://127.0.0.1:25610")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--remote-dir", default="/results/remote")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.phase == "populate":
        phase_populate(args)
    else:
        phase_retrieve(args)


if __name__ == "__main__":
    main()
