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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_TOKENS = 200
TEMPERATURE = 0.7


def run(model_path: str, prompts: list[str]) -> dict:
    from mlx_lm import load, generate

    model, tokenizer = load(model_path)
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
        resp = generate(
            model, tokenizer, prompt=full,
            max_tokens=MAX_TOKENS, temp=TEMPERATURE, verbose=False,
        )
        cases.append({"prompt": prompt, "output": resp.text})
    return {"model": model_path, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "cases": cases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    base = Path("models")
    targets = [args.model] if args.model else sorted(
        str(p) for p in base.glob("*mlx-bf16")
    ) + sorted(str(p) for p in base.glob("*mlx-4bit"))
    if not targets:
        raise SystemExit("No models found. Run steps 1-3 of the README first.")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    for target in targets:
        print(f"Testing {target} ...")
        data = run(target, prompts)
        path = results_dir / f"inference_{Path(target).name}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
