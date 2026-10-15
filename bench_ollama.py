"""Step 6b of the lab: benchmark models already downloaded in Ollama.

WHAT THIS IS FOR: you may already have models via `ollama list` (including
  Ollama's own `-mlx` builds). This script discovers them through Ollama's
  API, races each one with the same prompts/temperature-0 methodology as
  bench_engines.py, and writes one results/engines_ollama-<model>.json per
  model — which make_report.py picks up automatically in the race table.
INPUTS:  a running Ollama server (`ollama serve`; the desktop app works too).
OUTPUTS: results/engines_ollama-<model>.json (one file per model raced).

Usage:
    uv run python bench_ollama.py [--models qwen3.6:27b-mlx,gemma4:12b-mlx]
                                  [--repeats 3] [--base-url http://localhost:11434/v1]

Default (no --models): races every model Ollama reports as downloaded.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from bench_engines import print_table, stream_once  # same methodology as step 6
from labutil import complete, guarded

ROOT = Path(__file__).resolve().parent


def ollama_models(base_url: str) -> list[dict]:
    """List downloaded models via Ollama's API (server must be up)."""
    api = base_url.rstrip("/v1").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(api, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise SystemExit(
            f"Cannot reach Ollama at {api} ({e}). Start it with `ollama serve`."
        )
    return data.get("models", [])


def load_prompts() -> list[str]:
    return [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def safe_tag(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="Comma-separated subset (default: all downloaded).")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel in-flight requests (default: 1).")
    args = ap.parse_args()
    started = time.perf_counter()

    available = {m["name"]: m for m in ollama_models(args.base_url)}
    if not available:
        raise SystemExit("Ollama reports no downloaded models (`ollama list`).")
    wanted = ([m.strip() for m in args.models.split(",")] if args.models
              else sorted(available))
    missing = [m for m in wanted if m not in available]
    if missing:
        raise SystemExit(f"Not downloaded in Ollama: {missing}. "
                         f"Available: {sorted(available)}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    prompts = load_prompts()
    raced, skipped, wrote = [], [], []
    gpu = None  # one snapshot for the whole race (set per model below)
    for name in wanted:
        meta = available[name]
        # Models without chat (e.g. embed/decision-only) have no completions
        # endpoint — report and skip instead of failing the whole race.
        caps = meta.get("capabilities")
        if caps is not None and "completion" not in caps:
            print(f"Skipping {name}: no chat support (capabilities={caps}).")
            skipped.append(name)
            continue
        details = meta.get("details", {}) or {}
        print(f"Racing ollama:{name} "
              f"(runner={details.get('runner', '?')}, "
              f"size={round(meta.get('size', 0) / 1e9, 1)} GB) ...")
        from concurrent.futures import ThreadPoolExecutor
        from telemetry import gpu_snapshot, prime as tele_prime

        tele_prime()
        gpu = gpu_snapshot() if not raced else gpu  # one snapshot per race
        runs = []

        def one_timed(prompt: str) -> dict:
            # Per-run errors are recorded, not fatal (see bench_engines).
            try:
                r = stream_once(args.base_url, name, prompt)
                r["prompt"] = prompt[:80]
                return r
            except Exception as e:  # noqa: BLE001 — verbatim in JSON
                return {"prompt": prompt[:80],
                        "error": f"{type(e).__name__}: {e}",
                        "ttft_seconds": None, "ttft_to": None,
                        "total_seconds": None, "output_chars": None,
                        "chars_per_sec": None, "completion_tokens": None,
                        "prompt_tokens": None,
                        "tokens_per_sec": None, "itl_mean_ms": None,
                        "itl_p50_ms": None, "cpu_percent": None,
                        "mem_used_gb": None, "mem_percent": None}

        for prompt in prompts:
            stream_once(args.base_url, name, prompt)  # warmup, discarded
        jobs = [p for p in prompts for _ in range(args.repeats)]
        t_race = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            runs = list(ex.map(one_timed, jobs))
        makespan = time.perf_counter() - t_race
        out = {"engine_tag": "ollama", "base_url": args.base_url,
               "model": name, "repeats": args.repeats,
               "concurrency": args.concurrency,
               "makespan_s": round(makespan, 2),
               "runner": details.get("runner"),
               "size_gb": round(meta.get("size", 0) / 1e9, 1),
               "quantization": details.get("quantization_level"),
               "gpu": gpu, "runs": runs}
        path = results_dir / f"engines_ollama-{safe_tag(name)}.json"
        path.write_text(json.dumps(out, indent=2))
        raced.append(name)
        wrote.append(path)
        print(f"  wrote {path}")
        print_table("ollama", name, runs, makespan, args.concurrency,
                    str(path))
    all_runs = [r for p in wrote for r in json.loads(p.read_text())["runs"]]
    n_err = sum(1 for r in all_runs if r.get("error"))
    details = [f"{len(raced)} model(s) raced, {len(all_runs) - n_err}/{len(all_runs)} runs ok"
               + (f" ({n_err} errored, see JSON)" if n_err else ""),
               f"results: {', '.join(str(p) for p in wrote)}"]
    if skipped:
        details.append(f"skipped (no chat): {', '.join(skipped)}")
    complete("Ollama race", details, started,
             next_step="uv run python make_report.py")


if __name__ == "__main__":
    guarded(main)
