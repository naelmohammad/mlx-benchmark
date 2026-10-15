"""Step 4 of the lab: qualitative inference test (the quality gate).

WHAT THIS IS FOR: answers the question benchmarks cannot — "did 4-bit break
  coherence?" Asks every prompt in prompts.txt on each build and saves full
  transcripts for you to READ. Speed numbers are meaningless until this passes.
INPUTS:  prompts.txt; TEMPERATURE below (0.7 = natural chat variety).
OUTPUTS: results/inference_<model-dir-name>.json (prompt + output pairs).

Usage:
    uv run python inference_test.py [--model models/Qwen3-8B-mlx-4bit]

Default (no --model): tests the bf16 AND 4-bit pair side by side.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from labutil import complete, guarded

ROOT = Path(__file__).resolve().parent
MAX_TOKENS = 200
TEMPERATURE = 0.7


def run(model_path: str, prompts: list[str]) -> dict:
    # NOTE: mlx-lm >= 0.26 removed generate()'s sampler kwargs and made it
    # return plain text. stream_generate yields GenerationResponse objects
    # (text segments + metrics), so we collect text and keep the last
    # response. Temperature goes through make_sampler, not a kwarg.
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=TEMPERATURE)
    apply_chat = hasattr(tokenizer, "apply_chat_template")
    cases = []
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
        text = ""
        for resp in stream_generate(
            model, tokenizer, prompt=full,
            max_tokens=MAX_TOKENS, sampler=sampler,
        ):
            text += resp.text
        cases.append({"prompt": prompt, "output": text})
    return {"model": model_path, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "cases": cases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--list", action="store_true",
                    help="List discoverable local builds and exit.")
    args = ap.parse_args()
    started = time.perf_counter()

    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    base = Path("models")
    targets = [args.model] if args.model else sorted(
        str(p) for p in base.glob("*mlx-bf16")
    ) + sorted(str(p) for p in base.glob("*mlx-4bit"))
    if not targets:
        raise SystemExit("No models found in models/. Run steps 1-3 first, "
                         "or download a ready-made build (MODELS.md §3).")
    if args.list:
        print("Local MLX builds (default run tests all of these):")
        for t in targets:
            print(f"  {t}")
        return

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    wrote = []
    for target in targets:
        print(f"Testing {target} ...")
        data = run(target, prompts)
        path = results_dir / f"inference_{Path(target).name}.json"
        path.write_text(json.dumps(data, indent=2))
        wrote.append((path, len(data["cases"])))
        print(f"  wrote {path}")
    complete("Inference test",
             [f"{len(targets)} build(s), "
              f"{sum(n for _, n in wrote)} transcripts saved",
              f"results: {', '.join(str(p) for p, _ in wrote)}",
              "read the transcripts before trusting any tok/s number"],
             started, next_step="uv run python bench.py")


if __name__ == "__main__":
    guarded(main)
