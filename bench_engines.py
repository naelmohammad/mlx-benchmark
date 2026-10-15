"""Step 6 of the lab: race OpenAI-compatible servers (mlx-lm vs Rapid-MLX vs …).

WHAT THIS IS FOR: server-vs-server comparison with identical prompts. Streams
  /v1/chat/completions via SSE and measures TTFT (time-to-first-token: the
  snappiness you feel) plus decode rate. Temperature is fixed at 0.0 so every
  engine does byte-identical work — otherwise sampling noise pollutes the race.
HONESTY RULE: token counts come from the server's `usage` block when present;
  otherwise the script reports chars/sec and says so — it never invents tokens.
INPUTS:  a running server (--base-url), its served model name (--model).
OUTPUTS: results/engines_<tag>.json (one file per engine raced).

Usage (server already running in another terminal):
    uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \\
        --model "<served-model-name>" --tag mlx-lm [--repeats 3]
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
    ttft_to = None
    chars = 0
    completion_tokens = None
    # SSE stream: each "data:" line is one JSON chunk; "[DONE]" ends it.
    # The clock starts before the request so TTFT includes network + prefill.
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
            delta = choice.get("delta") or {}
            # Thinking models (Ollama) stream `reasoning` deltas with empty
            # content first: TTFT marks the first output of either kind, and
            # ttft_to records which, so cross-engine numbers stay honest.
            text = delta.get("content") or ""
            think = delta.get("reasoning") or ""
            if (text or think) and first_token_at is None:
                first_token_at = time.perf_counter()
                ttft_to = "content" if text else "reasoning"
            chars += len(text)
            # Some servers omit `usage` on streams (requested via
            # stream_options.include_usage above); then we fall back to chars/sec.
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
    total = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at else None
    decode_s = (total - ttft) if ttft else total
    return {
        "ttft_seconds": round(ttft, 3) if ttft is not None else None,
        "ttft_to": ttft_to,
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
    # A refused connection means the server isn't up — say so plainly instead
    # of dumping a urllib traceback (README step 6 needs the server in
    # another terminal first).
    try:
        for prompt in prompts:
            stream_once(args.base_url, args.model, prompt)  # warmup, discarded
            for _ in range(args.repeats):
                r = stream_once(args.base_url, args.model, prompt)
                r["prompt"] = prompt[:80]
                runs.append(r)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Cannot reach a server at {args.base_url} ({e.reason}).\n"
            f"Start it first, e.g. terminal 1:\n"
            f"  uv run mlx_lm.server --model models/<your>-mlx-4bit --port 8000\n"
            f"then re-run this command."
        )
    out = {"engine_tag": args.tag, "base_url": args.base_url,
           "model": args.model, "repeats": args.repeats, "runs": runs}
    path = results_dir / f"engines_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
