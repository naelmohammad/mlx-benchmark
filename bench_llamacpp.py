"""Step 6c of the lab: benchmark a llama.cpp server (llama-server).

WHAT THIS IS FOR: llama.cpp is the engine under Ollama's hood, run bare —
  no wrapper, every knob explicit. Same prompts/temperature-0/streamed-TTFT
  methodology as bench_engines (shared race_at), tagged `llama-cpp`, so its
  rows sit next to mlx-lm / Rapid-MLX / Ollama in the report race table.
INPUTS:  a GGUF file (see MODELS.md §5C for download) and a running server:
    llama-server -m models/gguf/<file>.gguf -c 8192 --port 8001
OUTPUTS: results/engines_llama-cpp-<model>.json (picked up by make_report).

Usage:
    uv run python bench_llamacpp.py --list                     # find local GGUFs
    uv run python bench_llamacpp.py --base-url http://127.0.0.1:8001/v1 \\
        --model <file.gguf> [--repeats 3] [--concurrency N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench_engines import print_table, race_at, stream_once
from labutil import complete, guarded

ROOT = Path(__file__).resolve().parent


def find_ggufs() -> list[Path]:
    return sorted(ROOT.glob("models/**/*.gguf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default=None,
                    help="Served model name (usually the .gguf filename).")
    ap.add_argument("--tag", default="llama-cpp")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--list", action="store_true",
                    help="List local GGUF files and exit.")
    args = ap.parse_args()
    started = time.perf_counter()

    ggufs = find_ggufs()
    if args.list:
        print("Local GGUF files (serve one, then race it):")
        for g in ggufs or ["(none — see MODELS.md §5C to download one)"]:
            print(f"  {g}")
            if ggufs:
                print("    start: "
                      f"llama-server -m {g} -c 8192 --port 8001")
        return
    if not args.model:
        raise SystemExit("Pass --model <served name>, or --list to discover "
                         "local GGUF files first.")

    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    try:
        stream_once(args.base_url, args.model, prompts[0])  # server-up check
    except Exception as e:  # noqa: BLE001 — actionable hint below
        raise SystemExit(
            f"Cannot race {args.base_url} ({e}).\n"
            f"Start it first, e.g.:\n"
            f"  llama-server -m models/gguf/<file>.gguf -c 8192 --port 8001")
    for prompt in prompts[1:]:
        stream_once(args.base_url, args.model, prompt)  # warmup, discarded
    from telemetry import gpu_snapshot, prime as tele_prime

    tele_prime()
    runs, makespan = race_at(args.base_url, args.model, prompts,
                             args.repeats, args.concurrency)
    out = {"engine_tag": args.tag, "base_url": args.base_url,
           "model": args.model, "repeats": args.repeats,
           "concurrency": args.concurrency,
           "makespan_s": round(makespan, 2),
           "gpu": gpu_snapshot(), "runs": runs}
    from bench_ollama import safe_tag

    path = results_dir / f"engines_{args.tag}-{safe_tag(args.model)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print_table(args.tag, args.model, runs, makespan, args.concurrency,
                str(path))
    ok = sum(1 for r in runs if not r.get("error"))
    complete(f"llama.cpp race ({args.model})",
             [f"{ok}/{len(runs)} runs ok", f"result: {path}"],
             started, next_step="uv run python make_report.py")


if __name__ == "__main__":
    guarded(main)
