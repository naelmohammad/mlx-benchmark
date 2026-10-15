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

from labutil import complete, guarded

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
    from charts import pct
    from telemetry import sample as tele_sample

    t0 = time.perf_counter()
    first_token_at = None
    ttft_to = None
    chars = 0
    reasoning_chars = 0
    completion_tokens = None
    arrivals: list[float] = []
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
            if text or think:
                # ITL is measured per output-bearing chunk — an approximation
                # of per-token gaps (servers usually stream token-per-chunk).
                # Thinking models emit `reasoning` chunks with empty content.
                arrivals.append(time.perf_counter())
            if (text or think) and first_token_at is None:
                first_token_at = time.perf_counter()
                ttft_to = "content" if text else "reasoning"
            chars += len(text)
            reasoning_chars += len(think)
            # Some servers omit `usage` on streams (requested via
            # stream_options.include_usage above); then we fall back to chars/sec.
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
    total = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at else None
    decode_s = (total - ttft) if ttft else total
    gaps_ms = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    tele = tele_sample()
    return {
        "ttft_seconds": round(ttft, 3) if ttft is not None else None,
        "ttft_to": ttft_to,
        "total_seconds": round(total, 3),
        "output_chars": chars,
        "reasoning_chars": reasoning_chars,
        "chars_per_sec": round(chars / decode_s, 1) if decode_s > 0 else None,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": (
            round(completion_tokens / decode_s, 1)
            if completion_tokens and decode_s > 0 else None
        ),
        "itl_mean_ms": round(sum(gaps_ms) / len(gaps_ms), 1) if gaps_ms else None,
        "itl_p50_ms": pct(gaps_ms, 50) if gaps_ms else None,
        # TPOT (time per output token): the per-token cost view of the same
        # decode. TPS answers "how many", TPOT "how long each".
        "tpot_ms": (round(1000 / (completion_tokens / decode_s), 2)
                    if completion_tokens and decode_s > 0 else None),
        "cpu_percent": tele["cpu_percent"],
        "mem_used_gb": tele["mem_used_gb"],
        "mem_percent": tele["mem_percent"],
    }


def col_stats(vals: list) -> list[str]:
    """avg/min/max/p99/p90/p50/std for one metric column (aiperf layout)."""
    import statistics

    from charts import pct

    vals = [v for v in vals if v is not None]
    if not vals:
        return ["N/A"] * 7
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return [f"{statistics.mean(vals):,.2f}", f"{min(vals):,.2f}",
            f"{max(vals):,.2f}", f"{pct(vals, 99):,.2f}",
            f"{pct(vals, 90):,.2f}", f"{pct(vals, 50):,.2f}", f"{std:,.2f}"]


def print_table(tag: str, model: str, runs: list[dict], makespan: float,
                concurrency: int, export_path: str) -> None:
    """aiperf-style console summary: the core four + request stats."""
    import sys as _sys

    from rich.console import Console
    from rich.table import Table

    ok = [r for r in runs if not r.get("error")]
    table = Table(title=f"{tag} : {model}  (concurrency={concurrency})",
                  show_header=True, header_style="bold")
    table.add_column("Metric", justify="right")
    for h in ("avg", "min", "max", "p99", "p90", "p50", "std"):
        table.add_column(h, justify="right")
    # The core four: TTFT, ITL, request latency, output token throughput.
    table.add_row("Time to First Token (ms)",
                  *col_stats([r["ttft_seconds"] * 1000 for r in ok]))
    table.add_row("Inter-Token Latency (ms)",
                  *col_stats([r["itl_mean_ms"] for r in ok]))
    table.add_row("Request Latency (ms)",
                  *col_stats([r["total_seconds"] * 1000 for r in ok]))
    table.add_row("Output Token Throughput (tokens/sec)",
                  *col_stats([r["tokens_per_sec"] for r in ok]))
    n = len(ok)
    tot_tok = sum(r["completion_tokens"] or 0 for r in ok)
    agg = f"{(tot_tok / makespan):,.2f}" if makespan > 0 and n else "N/A"
    table.add_row("Output Token Throughput, aggregate (tokens/sec)",
                  agg, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
    table.add_row("Request Throughput (requests/sec)",
                  f"{(n / makespan):,.2f}" if makespan > 0 else "N/A",
                  "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
    table.add_row("Request Count (requests)", str(n),
                  "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
    Console().print(table)
    print(f"CLI Command: {' '.join(_sys.argv)}")
    print(f"Benchmark Duration: {makespan:.2f} sec  |  Concurrency: "
          f"{concurrency}  |  Errors: {len(runs) - n}")
    print(f"JSON Export: {export_path}")


def race_at(base_url: str, model: str, prompts: list[str], repeats: int,
            concurrency: int) -> tuple[list[dict], float]:
    """One race at one concurrency level. Returns (runs, makespan_s)."""
    from concurrent.futures import ThreadPoolExecutor

    def one_timed(prompt: str) -> dict:
        # Per-run errors are recorded, not fatal: one bad response (timeout,
        # mid-run server hiccup) must not erase the runs that succeeded.
        try:
            r = stream_once(base_url, model, prompt)
            r["prompt"] = prompt[:80]
            return r
        except Exception as e:  # noqa: BLE001 — recorded verbatim below
            return {"prompt": prompt[:80], "error": f"{type(e).__name__}: {e}",
                    "ttft_seconds": None, "ttft_to": None,
                    "total_seconds": None, "output_chars": None,
                    "reasoning_chars": None, "chars_per_sec": None,
                    "completion_tokens": None, "tokens_per_sec": None,
                    "tpot_ms": None, "itl_mean_ms": None,
                    "itl_p50_ms": None, "cpu_percent": None,
                    "mem_used_gb": None, "mem_percent": None}

    jobs = [p for p in prompts for _ in range(repeats)]
    t_race = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        runs = list(ex.map(one_timed, jobs))
    return runs, time.perf_counter() - t_race


def frontier_point(runs: list[dict], makespan: float,
                   concurrency: int) -> dict:
    """One Pareto point: throughput vs latency at a concurrency level."""
    from charts import pct

    ok = [r for r in runs if not r.get("error")]
    lat = sorted(r["total_seconds"] * 1000 for r in ok
                 if r.get("total_seconds") is not None)
    itls = [r["itl_mean_ms"] for r in ok if r.get("itl_mean_ms") is not None]
    toks = sum(r["completion_tokens"] or 0 for r in ok)
    return {"concurrency": concurrency,
            "agg_tps": round(toks / makespan, 1) if makespan > 0 and ok else None,
            "rps": round(len(ok) / makespan, 2) if makespan > 0 else None,
            "p50_latency_ms": pct(lat, 50) if lat else None,
            "itl_p50_ms": pct(itls, 50) if itls else None,
            "errors": len(runs) - len(ok)}


def print_frontier(tag: str, model: str, frontier: list[dict]) -> None:
    """The Pareto frontier: throughput gained per latency paid, by load."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Pareto frontier : {tag} : {model} "
                        "(higher TPS vs lower latency)",
                  show_header=True, header_style="bold")
    for h in ("concurrency", "agg TPS", "RPS", "p50 latency ms", "ITL p50 ms",
              "errors"):
        table.add_column(h, justify="right")
    for p in frontier:
        table.add_row(str(p["concurrency"]), str(p["agg_tps"]), str(p["rps"]),
                      str(p["p50_latency_ms"]), str(p["itl_p50_ms"]),
                      str(p["errors"]))
    Console().print(table)
    print("Read it: TPS should climb with concurrency, then flatten (the "
          "saturation point) while latency climbs — past the knee, extra load "
          "buys throughput with latency. That knee is your capacity answer.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="e.g. mlx-lm or rapid-mlx")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel in-flight requests (default: 1).")
    ap.add_argument("--sweep", default=None,
                    help="Pareto sweep, e.g. --sweep 1,2,4: race each level "
                         "and report the throughput/latency frontier.")
    args = ap.parse_args()
    started = time.perf_counter()

    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    runs = []
    # Server-down check up front (warmup): a refused connection means no
    # server is listening; a 404 means the --model name doesn't match the
    # server's served model (mlx_lm.server wants the exact startup string).
    # NOTE: keep messages paste-safe — no <angle-bracket> placeholders,
    # zsh treats them as redirects.
    try:
        stream_once(args.base_url, args.model, prompts[0])  # warmup, discarded
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"Server at {args.base_url} answered {e.code} for model "
            f"'{args.model}'.\n"
            f"--model must equal the server's served name exactly:\n"
            f"  mlx_lm.server wants its full --model path, e.g.\n"
            f"  uv run python bench_engines.py --base-url {args.base_url} "
            f"--model models/Qwen3-8B-mlx-4bit --tag mlx-lm"
        )
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Cannot reach a server at {args.base_url} ({e.reason}).\n"
            f"Start a server first (separate terminal), then re-run:\n"
            f"  uv run mlx_lm.server --model models/Qwen3-8B-mlx-4bit --port 8000"
        )
    from telemetry import gpu_snapshot, prime as tele_prime

    tele_prime()  # seed CPU baseline before the timed phase
    gpu = gpu_snapshot()  # one snapshot per race (sudo-gated, else a reason)
    # Warmups stay sequential (one per prompt); timed runs go through
    # race_at's pool so --concurrency N measures real overlap.
    for prompt in prompts[1:]:
        stream_once(args.base_url, args.model, prompt)  # warmup, discarded
    levels = ([int(c) for c in args.sweep.split(",")] if args.sweep
              else [args.concurrency])
    # One file per level keeps runs comparable; the sweep file holds the
    # Pareto frontier across levels.
    all_runs, frontier = [], []
    for level in levels:
        runs, makespan = race_at(args.base_url, args.model, prompts,
                                 args.repeats, level)
        errors = sum(1 for r in runs if r.get("error"))
        out = {"engine_tag": args.tag, "base_url": args.base_url,
               "model": args.model, "repeats": args.repeats,
               "concurrency": level, "makespan_s": round(makespan, 2),
               "gpu": gpu, "runs": runs}
        suffix = f"{args.tag}-c{level}" if args.sweep else args.tag
        path = results_dir / f"engines_{suffix}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        if args.sweep:
            print(f"--- concurrency {level}: "
                  f"{len(runs) - errors}/{len(runs)} ok ---")
        else:
            print_table(args.tag, args.model, runs, makespan, level,
                        str(path))
        all_runs.extend(runs)
        frontier.append(frontier_point(runs, makespan, level))
    if args.sweep:
        fpath = results_dir / f"engines_{args.tag}-frontier.json"
        fpath.write_text(json.dumps(
            {"engine_tag": args.tag, "model": args.model,
             "levels": args.sweep, "frontier": frontier}, indent=2))
        print(f"wrote {fpath}")
        print_frontier(args.tag, args.model, frontier)
    ok = sum(1 for r in all_runs if not r.get("error"))
    complete(f"Engine race ({args.tag})",
             [f"{ok}/{len(all_runs)} runs ok over {len(prompts)} prompt(s)"
              + (f" ({len(all_runs) - ok} errored, see JSON)"
                 if len(all_runs) - ok else ""),
              f"levels raced: {levels}"],
             started, next_step="uv run python make_report.py")


if __name__ == "__main__":
    guarded(main)
