"""Step 6d of the lab: benchmark a Rapid-MLX server.

WHAT THIS IS FOR: Rapid-MLX claims up to 4× Apple's mlx-lm on the same
  weights. This races its OpenAI-compatible server with the shared
  methodology (race_at from bench_engines: same prompts, temperature 0,
  streamed TTFT), tagged `rapid-mlx` for the report race table.
INPUTS:  a running server:  rapid-mlx serve <alias>   (default :8000, or
    :8001 when 8000 is taken — match --base-url to what it reports).
    Discover aliases in the Rapid-MLX docs/model list (e.g. qwen3.5-9b-4bit).
OUTPUTS: results/engines_rapid-mlx-<alias>.json (picked up by make_report).

Usage:
    uv run python bench_rapid.py --alias qwen3.5-9b-4bit [--port 8001]
                                 [--repeats 3] [--concurrency N] [--sweep 1,2,4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench_engines import (frontier_point, print_frontier, print_table,
                           race_at, stream_once)
from bench_ollama import safe_tag
from labutil import complete, guarded

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True,
                    help="Rapid-MLX model alias, e.g. qwen3.5-9b-4bit.")
    ap.add_argument("--port", type=int, default=8000,
                    help="Port rapid-mlx serve reported (8001 if 8000 busy).")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--sweep", default=None,
                    help="Pareto sweep, e.g. --sweep 1,2,4.")
    args = ap.parse_args()
    started = time.perf_counter()

    base_url = f"http://localhost:{args.port}/v1"
    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    try:
        stream_once(base_url, args.alias, prompts[0])  # server-up check
    except Exception as e:  # noqa: BLE001 — actionable hint below
        raise SystemExit(
            f"Cannot race {base_url} ({e}).\n"
            f"Start it first (separate terminal):\n"
            f"  rapid-mlx serve {args.alias}")
    for prompt in prompts[1:]:
        stream_once(base_url, args.alias, prompt)  # warmup, discarded
    from telemetry import gpu_snapshot, prime as tele_prime

    tele_prime()
    gpu = gpu_snapshot()
    levels = ([int(c) for c in args.sweep.split(",")] if args.sweep
              else [args.concurrency])
    frontier = []
    for level in levels:
        runs, makespan = race_at(base_url, args.alias, prompts,
                                 args.repeats, level)
        out = {"engine_tag": "rapid-mlx", "base_url": base_url,
               "model": args.alias, "repeats": args.repeats,
               "concurrency": level, "makespan_s": round(makespan, 2),
               "gpu": gpu, "runs": runs}
        suffix = f"rapid-mlx-{safe_tag(args.alias)}" + (
            f"-c{level}" if args.sweep else "")
        path = results_dir / f"engines_{suffix}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        if args.sweep:
            print(f"--- concurrency {level}: "
                  f"{sum(1 for r in runs if not r.get('error'))}/{len(runs)} ok ---")
        else:
            print_table("rapid-mlx", args.alias, runs, makespan, level,
                        str(path))
        frontier.append(frontier_point(runs, makespan, level))
    if args.sweep:
        fpath = (results_dir /
                 f"engines_rapid-mlx-{safe_tag(args.alias)}-frontier.json")
        fpath.write_text(json.dumps(
            {"engine_tag": "rapid-mlx", "model": args.alias,
             "levels": args.sweep, "frontier": frontier}, indent=2))
        print(f"wrote {fpath}")
        print_frontier("rapid-mlx", args.alias, frontier)
    complete("Rapid-MLX race",
             [f"levels raced: {levels}"],
             started, next_step="uv run python make_report.py")


if __name__ == "__main__":
    guarded(main)
