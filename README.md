# MLX benchmark lab

Learn the full local-LLM loop on Apple Silicon, then verify vendor speed
claims yourself: convert and quantize a Hugging Face model to MLX, benchmark
Apple's `mlx-lm`, and race it against
[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) over an OpenAI-compatible
endpoint. Same prompts, same machine, your own numbers.

Default model is small on purpose (`Qwen/Qwen3-8B`, ~16 GB download) so the
whole loop finishes in one sitting. Scale up with `MODEL_ID` once you trust
the pipeline.

## Prerequisites

- Apple Silicon Mac, Python 3.12+, `uv`, `git`, `curl`
- ~40 GB free disk (HF weights + bf16 MLX + 4-bit MLX for the 8B default)
- A [Hugging Face token](https://huggingface.co/settings/tokens) only for
  gated models (Qwen3-8B is public — no token needed)

## Setup

```bash
uv venv --python 3.12
uv sync
```

(`uv run` below uses that project-local `.venv` automatically.)

## The pipeline, step by step

Each step is a script or one command. Run them in order the first time so you
see what every stage does.

### 1. Download the weights from Hugging Face

```bash
uv run huggingface-cli download "$MODEL_ID" --local-dir "models/$(basename $MODEL_ID)-hf"
# default MODEL_ID=Qwen/Qwen3-8B; override per-command, e.g. MODEL_ID=Qwen/Qwen3-30B-A3B
```

What you learn: repos are just versioned folders of safetensors + tokenizer
+ config. Check size with `du -sh models/*`.

### 2. Convert HF format to MLX (bf16, no quantization yet)

```bash
uv run mlx_lm.convert \
  --hf-path "models/$(basename $MODEL_ID)-hf" \
  --mlx-path "models/$(basename $MODEL_ID)-mlx-bf16"
```

What you learn: MLX stores weights as `.npy` shards plus `config.json`.
Compare `du -sh` of the HF dir vs the MLX dir — same information, new layout.

### 3. Quantize a 4-bit copy

```bash
uv run mlx_lm.convert \
  --hf-path "models/$(basename $MODEL_ID)-mlx-bf16" \
  --mlx-path "models/$(basename $MODEL_ID)-mlx-4bit" \
  -q --q-bits 4
```

What you learn: quantization trades precision for size/speed. Note the
bf16 → 4-bit size ratio; you will feel it again in the benchmark numbers.

### 4. Smoke-test inference

```bash
uv run mlx_lm.generate --model "models/$(basename $MODEL_ID)-mlx-4bit" \
  --prompt "Explain quantization in one paragraph." --max-tokens 150
```

Then run the scripted test, which asks every prompt in `prompts.txt` on both
the bf16 and 4-bit builds and saves the transcripts:

```bash
uv run python inference_test.py
```

What you learn: this is your qualitative check — does 4-bit still answer
coherently? Read `results/inference_*.json` before trusting any tok/s number.

### 5. Benchmark Apple's mlx-lm (in-process)

```bash
uv run python bench.py
```

Measures, per prompt, after one warmup run: time-to-first-token, prompt
(prefill) tok/s, generation (decode) tok/s, and peak unified-memory used.
Writes `results/bench_<model>_<precision>.json` for bf16 and 4-bit.

### 6. Race mlx-lm against Rapid-MLX (OpenAI-compatible servers)

Terminal 1 — serve your converted 4-bit model with Apple's server:

```bash
uv run mlx_lm.server --model "models/$(basename $MODEL_ID)-mlx-4bit" --port 8000
```

Terminal 2 — benchmark any OpenAI-compatible endpoint with the same prompts:

```bash
uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \
  --model "$(basename $MODEL_ID)-mlx-4bit" --tag mlx-lm
```

Terminal 3 — serve Rapid-MLX (install once: `brew install rapid-mlx`):

```bash
rapid-mlx serve qwen3.5-9b-4bit   # listens on http://localhost:8000/v1 by default
```

```bash
uv run python bench_engines.py --base-url http://localhost:8000/v1 \
  --model "qwen3.5-9b-4bit" --tag rapid-mlx
```

Methodology note, read this before quoting numbers: this compares
*same-class* models (your 8B Qwen3 vs Rapid's 9B Qwen3.5 alias), not identical
weights — Rapid-MLX serves its own aliases, so identical-weights A/B is only
possible if it ever accepts a local `--model` path (check
`rapid-mlx serve --help`). Same prompts, same machine, streaming TTFT +
decode rate: still a fair server-vs-server race. Rapid-MLX also ships its own
fixed-protocol benchmark (`rapid-mlx benchmark run <alias>`); that number is
*not* comparable to live-serving numbers, including ours.

### 7. Generate the report

```bash
uv run python make_report.py
```

Merges every `results/*.json` into `REPORT.md`: machine specs, model sizes on
disk, bf16-vs-4bit quality/speed deltas, and the mlx-lm vs Rapid-MLX table.

### One-command rerun

```bash
./run_all.sh   # full mlx-lm pipeline (steps 1–5 + 7); engines compared separately
```

## Results

Results are yours and gitignored (`results/*.json`, `models/`). Commit only
`REPORT.md` when you want to publish a run.

## FAQ for learners

- **Ollama vs MLX?** Recent Ollama uses an MLX backend for some models on
  Apple Silicon, but its model files are GGUF. Here you work with MLX-native
  weights directly — no translation layer.
- **Why is decode tok/s the headline number?** Prefill is one parallel pass;
  decode generates one token at a time and dominates wall-clock for chat.
- **Why warmup?** First run pays kernel-compilation and cache costs. We
  discard it so the numbers reflect steady state.
