"""Race OpenAI-compatible endpoints with identical prompts (server-vs-server).

Usage (server already running in another terminal):
    uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \\
        --model "<served-model-name>" --tag mlx-lm [--repeats 3]

Streams chat completions and measures time-to-first-token plus output rate.
Token counts come from the server's `usage` block when present; otherwise the
script reports chars/sec and says so — it never invents token counts.
Writes results/engines_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stream_once(base_url: str, model: str, prompt: str) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,  # deterministic: same workload for every engine
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_token_at = None
    chars = 0
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = (choice.get("delta") or {}).get("content") or ""
            if delta and first_token_at is None:
                first_token_at = time.perf_counter()
            chars += len(delta)
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
    total = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at else None
    decode_s = (total - ttft) if ttft else total
    return {
        "ttft_seconds": round(ttft, 3) if ttft is not None else None,
        "total_seconds": round(total, 3),
        "output_chars": chars,
        "chars_per_sec": round(chars / decode_s, 1) if decode_s > 0 else None,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": (
            round(completion_tokens / decode_s, 1)
            if completion_tokens and decode_s > 0 else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="e.g. mlx-lm or rapid-mlx")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    runs = []
    for prompt in prompts:
        stream_once(args.base_url, args.model, prompt)  # warmup, discarded
        for _ in range(args.repeats):
            r = stream_once(args.base_url, args.model, prompt)
            r["prompt"] = prompt[:80]
            runs.append(r)
    out = {"engine_tag": args.tag, "base_url": args.base_url,
           "model": args.model, "repeats": args.repeats, "runs": runs}
    path = results_dir / f"engines_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
