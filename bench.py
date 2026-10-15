"""Step 5 of the lab: benchmark an MLX model in-process.

WHAT THIS IS FOR: measures the three numbers that describe inference speed —
  TTFT is approximated here by prefill rate; prompt_tps = prefill (prompt
  processing) tok/s; generation_tps = decode tok/s, the headline number for
  chat. Also records peak unified-memory per run (weights + KV cache).
WHY WARMUP: the first run compiles Metal kernels, so it is discarded — without
  this you benchmark the compiler, not the model (see CONCEPTS.md).
INPUTS:  prompts.txt (one prompt per line), one or two converted model dirs.
OUTPUTS: results/bench_<model-dir-name>.json (one file per build benched).

Usage:
    uv run python bench.py [--model models/Qwen3-8B-mlx-4bit] [--repeats 3]

Default (no --model): benches the bf16 AND 4-bit pair so you can compare them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from labutil import complete, guarded

ROOT = Path(__file__).resolve().parent


def load_prompts() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "prompts.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def timed_run(model, tokenizer, full: str, max_tokens: int, sampler):
    # NOTE: mlx-lm >= 0.26: generate() returns plain text, so metrics must
    # come from stream_generate's final GenerationResponse (prompt_tps,
    # generation_tps, peak_memory). sampler fixes temperature (0.0 = greedy,
    # deterministic numbers).
    from mlx_lm import stream_generate  # local import: heavy, only needed here

    t0 = time.perf_counter()
    for resp in stream_generate(
        model, tokenizer, prompt=full, max_tokens=max_tokens, sampler=sampler
    ):
        pass
    return resp, time.perf_counter() - t0


def benchmark_model(model_path: str, prompts: list[str], repeats: int) -> dict:
    from mlx_lm import load  # local import: heavy, only needed here
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.0)
    out = {"model": model_path, "repeats": repeats, "runs": []}

    apply_chat = hasattr(tokenizer, "apply_chat_template")
    for prompt in prompts:
        full = prompt
        if apply_chat:
            try:
                full = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                full = prompt

        # Warmup (discarded): first run compiles Metal kernels, so it would
        # poison the measurement. Short max_tokens keeps it cheap.
        timed_run(model, tokenizer, full, 32, sampler)

        for _ in range(repeats):
            resp, dt = timed_run(model, tokenizer, full, 256, sampler)
            # getattr fallbacks: field names differ across mlx-lm versions;
            # a missing metric is recorded as null, never faked.
            # peak_memory units also differ (bytes vs GB): over 1000 must be bytes.
            peak = getattr(resp, "peak_memory", None) or 0
            peak_gb = round(peak / 1e9, 3) if peak > 1000 else (round(peak, 3) or None)
            out["runs"].append(
                {
                    "prompt": prompt[:80],
                    "prompt_tokens": getattr(resp, "prompt_tokens", None),
                    "prompt_tps": getattr(resp, "prompt_tps", None),
                    "generation_tokens": getattr(resp, "generation_tokens", None),
                    "generation_tps": getattr(resp, "generation_tps", None),
                    "peak_memory_gb": peak_gb,
                    "wall_seconds": round(dt, 3),
                }
            )
    return out


def shape_ids(tokenizer, pp: int) -> list[int]:
    """Exactly pp prompt token IDs: chat-wrapped seed text, padded with
    neutral filler (or truncated) so prefill cost is workload-fixed, omlx
    style (pp1024, pp4096, …). Token IDs go straight in — no detokenize
    round-trip that would change the count."""
    seed = "Explain attention in transformers in three sentences."
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            seed = tokenizer.apply_chat_template(
                [{"role": "user", "content": seed}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    ids = tokenizer.encode(seed)
    if len(ids) >= pp:
        return ids[:pp]
    filler = tokenizer.encode(" The quick brown fox jumps over the lazy dog.")
    while len(ids) < pp:
        ids += filler
    return ids[:pp]


def benchmark_shapes(model_path: str, shapes: list[tuple[int, int]],
                     repeats: int) -> dict:
    """Single-request matrix: fixed prompt tokens (pp) × fixed output (tg).

    Row semantics match omlx: TTFT = prefill time; TPOT = decode per token;
    PP TPS = prefill rate; TG TPS = decode rate; E2E = wall; Throughput =
    (pp + generated) / wall (total system utilization); Peak Mem observed.
    """
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.0)
    out = {"model": model_path, "repeats": repeats, "shapes": []}
    timed_run(model, tokenizer, shape_ids(tokenizer, 32), 8, sampler)  # warmup
    for pp, tg in shapes:
        ids = shape_ids(tokenizer, pp)
        for _ in range(repeats):
            resp, dt = timed_run(model, tokenizer, ids, tg, sampler)
            pt = getattr(resp, "prompt_tokens", None) or pp
            gt = getattr(resp, "generation_tokens", None) or 0
            ppt = getattr(resp, "prompt_tps", None)
            tgt = getattr(resp, "generation_tps", None)
            ttft_ms = round(pt / ppt * 1000, 1) if ppt else None
            tpot = round(1000 / tgt, 2) if tgt else None
            peak = getattr(resp, "peak_memory", None) or 0
            out["shapes"].append({
                "test": f"pp{pp}/tg{tg}",
                "ttft_ms": ttft_ms, "tpot_ms": tpot,
                "pp_tps": round(ppt, 1) if ppt else None,
                "tg_tps": round(tgt, 1) if tgt else None,
                "e2e_s": round(dt, 3),
                "throughput_tps": (round((pt + gt) / dt, 1)
                                   if dt > 0 else None),
                "peak_mem_gb": (round(peak / 1e9, 3) if peak > 1000
                                else (round(peak, 3) or None)),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single MLX dir to bench (default: every local build).")
    ap.add_argument("--list", action="store_true",
                    help="List discoverable local builds and exit.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shapes", default=None,
                    help="Single-request matrix, e.g. --shapes pp1024/tg128,"
                         "pp4096/tg128 (writes bench_shapes_<build>.json).")
    args = ap.parse_args()
    started = time.perf_counter()

    base = Path("models")
    targets = [args.model] if args.model else sorted(
        str(p) for p in base.glob("*mlx-bf16")
    ) + sorted(str(p) for p in base.glob("*mlx-4bit"))
    if not targets:
        raise SystemExit("No models found in models/. Run steps 1-3 first, "
                         "or download a ready-made build (MODELS.md §3).")
    if args.list:
        # Discovery mode: show what a default run would bench, then stop.
        print("Local MLX builds (default run benches all of these):")
        for t in targets:
            print(f"  {t}")
        return

    shapes = None
    if args.shapes:
        # e.g. "pp1024/tg128,pp4096/tg128" -> [(1024, 128), (4096, 128)]
        shapes = []
        for spec in args.shapes.split(","):
            pp, tg = spec.strip().lower().split("/tg")
            shapes.append((int(pp.lstrip("pp")), int(tg)))
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    prompts = load_prompts()
    wrote = []
    for target in targets:
        if shapes:
            print(f"Benchmarking shapes {args.shapes} on {target} ...")
            data = benchmark_shapes(target, shapes, args.repeats)
            path = results_dir / f"bench_shapes_{Path(target).name}.json"
            n = len(data["shapes"])
        else:
            print(f"Benchmarking {target} ...")
            data = benchmark_model(target, prompts, args.repeats)
            path = results_dir / f"bench_{Path(target).name}.json"
            n = len(data["runs"])
        path.write_text(json.dumps(data, indent=2))
        wrote.append((path, n))
        print(f"  wrote {path}")
    total_runs = sum(n for _, n in wrote)
    complete("Benchmark" + (" (shapes)" if shapes else ""),
             [f"{len(targets)} build(s), {total_runs} timed run(s)",
              f"results: {', '.join(str(p) for p, _ in wrote)}"],
             started, next_step="uv run python make_report.py")


if __name__ == "__main__":
    guarded(main)
