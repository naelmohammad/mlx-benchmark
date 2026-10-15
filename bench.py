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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single MLX dir to bench (default: bf16 + 4bit pair).")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    base = Path("models")
    targets = [args.model] if args.model else sorted(
        str(p) for p in base.glob("*mlx-bf16")
    ) + sorted(str(p) for p in base.glob("*mlx-4bit"))
    if not targets:
        raise SystemExit("No models found. Run steps 1-3 of the README first.")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    prompts = load_prompts()
    for target in targets:
        print(f"Benchmarking {target} ...")
        data = benchmark_model(target, prompts, args.repeats)
        name = Path(target).name.replace("models", "").strip("/-") or "model"
        path = results_dir / f"bench_{Path(target).name}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
