"""Benchmark an MLX model in-process: TTFT, prefill tok/s, decode tok/s, peak memory.

Usage:
    uv run python bench.py [--model models/Qwen3-8B-mlx-4bit] [--repeats 3]

Runs every prompt in prompts.txt against the bf16 AND 4-bit builds (when both
exist), after one discarded warmup run. Writes results/bench_<name>.json.
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


def benchmark_model(model_path: str, prompts: list[str], repeats: int) -> dict:
    from mlx_lm import load, generate  # local import: heavy, only needed here

    model, tokenizer = load(model_path)
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

        # Warmup (discarded): pays kernel-compile + cache costs.
        generate(model, tokenizer, prompt=full, max_tokens=32, verbose=False)

        for _ in range(repeats):
            t0 = time.perf_counter()
            resp = generate(
                model, tokenizer, prompt=full, max_tokens=256, verbose=False
            )
            dt = time.perf_counter() - t0
            out["runs"].append(
                {
                    "prompt": prompt[:80],
                    "prompt_tokens": getattr(resp, "prompt_tokens", None),
                    "prompt_tps": getattr(resp, "prompt_tps", None),
                    "generation_tokens": getattr(resp, "generation_tokens", None),
                    "generation_tps": getattr(resp, "generation_tps", None),
                    "peak_memory_gb": (
                        round(getattr(resp, "peak_memory", 0) / 1e9, 3)
                        if getattr(resp, "peak_memory", None)
                        else None
                    ),
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
