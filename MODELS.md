# Models: what to download, where it goes, and which runner to use

Every ID below was verified against the Hugging Face API. Sizes are bf16
download size; 4-bit MLX is roughly a quarter of that.

## 1. Catalog — source weights by vendor

| Vendor | Model | HF ID | bf16 size | Notes |
|---|---|---|---|---|
| Alibaba (Qwen) | Qwen3-8B (default) | `Qwen/Qwen3-8B` | ~16 GB | Small enough to learn the full loop in one sitting |
| Alibaba (Qwen) | Qwen3-32B | `Qwen/Qwen3-32B` | ~64 GB | Best quality that still fits 64 GB at 4-bit |
| Alibaba (Qwen) | Qwen3-30B-A3B (MoE) | `Qwen/Qwen3-30B-A3B` | ~60 GB | Only ~3B params active per token — fast |
| Alibaba (Qwen) | Qwen3.5-9B | `Qwen/Qwen3.5-9B` | ~18 GB | 2026 generation, 262K context |
| Alibaba (Qwen) | Qwen3.6-35B-A3B (MoE) | `Qwen/Qwen3.6-35B-A3B` | ~70 GB | Sweet spot for this machine; convert needs ~70 GB free |
| Alibaba (Qwen) | Qwen3-Coder-30B-A3B | `Qwen/Qwen3-Coder-30B-A3B` | ~60 GB | Gated — `huggingface-cli login` first; best open coding weights |
| Google | Gemma 3 27B IT | `google/gemma-3-27b-it` | ~54 GB | Gated — accept license + login; strong all-rounder |
| Meta | Llama 3.3 70B Instruct | `meta-llama/Llama-3.3-70B-Instruct` | ~140 GB | Gated; fits 64 GB only at 4-bit (~40 GB), tight |
| DeepSeek | R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | ~64 GB | Reasoning traces; weak at tool calling — benchmark, don't agent |
| OpenAI | gpt-oss-20b | `openai/gpt-oss-20b` | ~40 GB | Apache 2.0, built for agents/tool use |
| Alibaba (older) | Qwen2.5-32B Instruct | `Qwen/Qwen2.5-32B-Instruct` | ~64 GB | Battle-tested baseline to compare newer models against |

`HF_TOKEN` is required for these labs. How to generate one:

1. Sign up / log in at [huggingface.co](https://huggingface.co/join).
2. Open [Settings → Access Tokens](https://huggingface.co/settings/tokens).
3. Click **Create new token**, give it a name (e.g. `mlx-lab`), type **Read**,
   and create. (Fine-grained tokens work too — it only needs read access to
   public and gated repos you've been granted.)
4. Copy the token (it starts with `hf_`) — it is shown once.
5. For gated models (Llama, Gemma, Qwen-Coder), also open the model's HF
   page and click through its license agreement; access is usually granted
   instantly, and the download fails until then.

Export it before any download — the CLI sends it even for public models,
which avoids anonymous rate limits, and gated models refuse without it:

```bash
export HF_TOKEN="hf_..."
uv run huggingface-cli download <HF_ID> --local-dir "models/<name>-hf"
```

Or log in from the CLI instead of exporting the variable each time —
either form persists the token for subsequent commands:

```bash
uv run huggingface-cli login                 # interactive: paste token when asked
uv run huggingface-cli login --token "$HF_TOKEN"   # non-interactive equivalent
```

## 2. Shortcut — pre-quantized MLX ports (skip steps 2–3)

The community publishes ready-made 4-bit MLX builds. Download one and jump
straight to step 4 (inference test). You learn less, but you get a second
data point: compare *their* quant against *yours*.

| Port | HF ID |
|---|---|
| Qwen3-32B 4-bit | `mlx-community/Qwen3-32B-4bit` |
| Qwen3.5-9B 4-bit | `mlx-community/Qwen3.5-9B-MLX-4bit` |
| Qwen3.8-27B 4-bit | `mlx-community/Qwen3.8-27B-4bit` |
| Qwen3.6-35B-A3B 4-bit | `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` |
| Llama-3.3-70B 4-bit | `mlx-community/Llama-3.3-70B-Instruct-4bit` |

```bash
uv run huggingface-cli download <PORT_ID> --local-dir "models/<name>-mlx-4bit"
uv run python bench.py --model "models/<name>-mlx-4bit"
```

## 3. Where model locations are defined

Everything is an env var or a flag — no hardcoded paths:

| Knob | Default | Used by |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-8B` | `run_all.sh` (download/convert/quantize dirs derive from it) |
| `models/<short>-hf` | from `MODEL_ID` | HF download target (step 1) |
| `models/<short>-mlx-bf16` | from `MODEL_ID` | convert output (step 2), quantize input (step 3) |
| `models/<short>-mlx-4bit` | from `MODEL_ID` | quantize output (step 3); default bench target |
| `bench.py --model` | bf16 + 4-bit pair | Bench one dir instead of the pair |
| `bench_engines.py --base-url --model --tag` | (required) | Which server and served-model name to race |
| Rapid-MLX alias | `qwen3.5-9b-4bit` | README step 6 (`rapid-mlx serve <alias>`) |

Examples:

```bash
MODEL_ID=Qwen/Qwen3-30B-A3B ./run_all.sh   # whole pipeline on a bigger MoE
uv run python bench.py --model models/Qwen3-30B-A3B-mlx-4bit
```

## 4. Runner options — four ways to serve a model

**A. mlx-lm (this lab's path).** Convert HF → MLX yourself, serve with
`mlx_lm.server`, benchmark in-process. Maximum learning, Apple-native speed.
See README steps 1–5.

**B. Ollama (easiest).** One tool for download + serve + API. Recent Ollama
routes supported models through an MLX backend on Apple Silicon; weights are
GGUF, not MLX-native. Best when you want results, not lessons:

```bash
brew install ollama
ollama pull gpt-oss:20b     # library names: qwen3, qwen3.5, llama3.3, gemma3, deepseek-r1…
ollama run gpt-oss:20b
```

Ollama also exposes OpenAI-compatible endpoints, so `bench_engines.py` can
race it: serve with `ollama serve`, then point `--base-url` at
`http://localhost:11434/v1`. Compare its numbers against your MLX-native run
of the same-class model.

**C. llama.cpp (maximum control).** The engine under Ollama's hood, without
the wrapper. GGUF weights + explicit flags for every knob (threads, batch,
cache, quant):

```bash
brew install llama.cpp
uv run huggingface-cli download ggml-org/gpt-oss-20b-GGUF \
  --include "*.Q4_K_M.gguf" --local-dir models/gguf
llama-server -m models/gguf/*.gguf -c 8192 --port 8001
```

Then `bench_engines.py --base-url http://127.0.0.1:8001/v1 --tag llama-cpp`.
Use this when a number looks wrong and you want to isolate which layer (app,
server, engine) owns it.

**D. Rapid-MLX (the claim to test).** MLX-native server, OpenAI-compatible,
claims up to 4× Apple's mlx-lm on the same weights:

```bash
brew install rapid-mlx
rapid-mlx serve qwen3.5-9b-4bit        # localhost:8000, OpenAI-compatible
rapid-mlx benchmark run qwen3.5-9b-4bit  # their fixed protocol (not comparable to serving)
```

Race it with `bench_engines.py --tag rapid-mlx` per README step 6.

Rule of thumb: learn with A, daily-drive with B, debug with C, verify hype
with D.
