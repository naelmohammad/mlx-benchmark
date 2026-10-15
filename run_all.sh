#!/bin/bash
# run_all.sh — the whole mlx-lm pipeline in one command (README steps 1-5 + 7).
# WHAT IT DOES: download (HF) -> convert (bf16 MLX) -> quantize (4-bit) ->
#   inference transcripts -> benchmark -> REPORT.md. Override the model with
#   MODEL_ID=<hf-id> ./run_all.sh. The engine race (step 6) stays interactive
#   because it needs servers in separate terminals — see README step 6.
set -euo pipefail
cd "$(dirname "$0")"

: "${HF_TOKEN:?Set HF_TOKEN first: export HF_TOKEN=\"hf_...\" (see README Setup)}"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SHORT="$(basename "$MODEL_ID")"
HF_DIR="models/${SHORT}-hf"
BF16_DIR="models/${SHORT}-mlx-bf16"
Q4_DIR="models/${SHORT}-mlx-4bit"

echo "==> [0/6] environment"
uv venv --python 3.12 2>/dev/null || true
uv sync

echo "==> [1/6] download $MODEL_ID"
uv run hf download "$MODEL_ID" --local-dir "$HF_DIR"

echo "==> [2/6] convert to MLX (bf16)"
uv run mlx_lm.convert --hf-path "$HF_DIR" --mlx-path "$BF16_DIR"
du -sh "$HF_DIR" "$BF16_DIR"

echo "==> [3/6] quantize to 4-bit"
uv run mlx_lm.convert --hf-path "$BF16_DIR" --mlx-path "$Q4_DIR" -q --q-bits 4
du -sh "$Q4_DIR"

echo "==> [4/6] inference test (bf16 vs 4-bit transcripts)"
uv run python inference_test.py

echo "==> [5/6] benchmark (bf16 vs 4-bit)"
uv run python bench.py

echo "==> [6/6] report"
uv run python make_report.py
echo "Done. Read REPORT.md, then race Rapid-MLX per README step 6."
