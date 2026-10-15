# Models: what to download, where it goes, and which runner to use

Every ID below was verified against the Hugging Face API. Sizes are bf16
download size; 4-bit MLX is roughly a quarter of that.

## Quick pick — "I want…"

| You want… | Download this | Then… |
|---|---|---|
| The default learner loop | `Qwen/Qwen3-8B` | README steps 1–7 |
| Best quality on 64 GB | `Qwen/Qwen3-32B` | README steps 1–7 |
| Fastest big model (MoE) | `Qwen/Qwen3-30B-A3B` | README steps 1–7 |
| 2026 generation, small | `Qwen/Qwen3.5-4B` | README steps 1–7 |
| The 35B MoE showdown | `Qwen/Qwen3.5-35B-A3B` vs `Qwen/Qwen3.6-35B-A3B` | Worked example below |
| Coding with agents | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Serve it, point OpenClaw at it |
| Skip converting entirely | Any [pre-quantized port](#2-shortcut--pre-quantized-mlx-ports-skip-steps-23) | Jump to README step 4 |

## 1. Token setup (do once) ☑️

- [ ] HF account created
- [ ] Read token generated and saved
- [ ] Logged in via CLI or exported `HF_TOKEN`
- [ ] Gated-model licenses accepted (only for Llama, Gemma, Qwen-Coder)

`HF_TOKEN` is required for these labs — even public downloads send it (avoids
anonymous rate limits), and gated models refuse without it. Generate one:

1. Sign up / log in at [huggingface.co](https://huggingface.co/join).
2. Open [Settings → Access Tokens](https://huggingface.co/settings/tokens).
3. **Create new token**, name it (e.g. `mlx-lab`), type **Read**, create.
4. Copy the `hf_…` value — shown once.
5. For gated models, open the model's HF page and accept its license.

Then either export it per session or log in persistently:

```bash
export HF_TOKEN="hf_..."
uv run huggingface-cli download <HF_ID> --local-dir "models/<name>-hf"
# or persist login instead of exporting each time:
uv run huggingface-cli login                 # interactive paste
uv run huggingface-cli login --token "$HF_TOKEN"   # non-interactive
```

## 2. Catalog — source weights by family

<details open>
<summary><strong>Alibaba Qwen — dense chat models</strong> (click to collapse)</summary>

| Model | HF ID | bf16 | Notes |
|---|---|---|---|
| Qwen3-8B (default) | `Qwen/Qwen3-8B` | ~16 GB | Learn the full loop in one sitting |
| Qwen3-32B | `Qwen/Qwen3-32B` | ~64 GB | Best quality that fits 64 GB at 4-bit |
| Qwen3.5-2B / 4B | `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` | ~4 / ~8 GB | Small, fast 2026 gen; 2B runs anywhere |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | ~18 GB | 2026 gen, 262K context |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | ~54 GB | Mid-size 3.5 for quality runs |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` | ~54 GB | Dense 3.6; simpler alternative to the MoE |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | ~54 GB | Newest dense Qwen in the explorer — compare vs 3.6-27B |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | ~64 GB | Battle-tested baseline for newer models |

</details>

<details>
<summary><strong>Alibaba Qwen — MoE models (fast for their size)</strong></summary>

| Model | HF ID | bf16 | Notes |
|---|---|---|---|
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | ~60 GB | Only ~3B params active per token |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` | ~70 GB | MoE rival to 3.6-35B — benchmark them against each other |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | ~70 GB | Sweet spot for this machine |

An MoE (mixture-of-experts) holds many parameters but routes each token
through a few of them — big-model quality near small-model speed. That is why
a "35B" MoE can outrun a 27B dense model.

</details>

<details>
<summary><strong>Alibaba Qwen — coders</strong></summary>

| Model | HF ID | bf16 | Notes |
|---|---|---|---|
| Qwen3-Coder-30B-A3B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | ~60 GB | Best open coding weights; serve to OpenClaw/agents |
| Qwen2.5-Coder-32B | `Qwen/Qwen2.5-Coder-32B-Instruct` | ~64 GB | Previous gen, still strong |
| Qwen2.5-Coder-14B | `Qwen/Qwen2.5-Coder-14B-Instruct` | ~28 GB | Mid-size coder |
| Qwen2.5-Coder-7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | ~14 GB | The quick learner loop for code |

</details>

<details>
<summary><strong>Google, Meta, DeepSeek, OpenAI</strong></summary>

| Model | HF ID | bf16 | Notes |
|---|---|---|---|
| Gemma 3 27B IT | `google/gemma-3-27b-it` | ~54 GB | Gated — accept license + login |
| Gemma 4 26B-A4B IT (MoE) | `google/gemma-4-26B-A4B-it` | ~52 GB | 2026 gen; ~4B active — fast; login if asked |
| Gemma 4 31B IT | `google/gemma-4-31B-it` | ~62 GB | 2026 gen dense; login if asked |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | ~140 GB | Gated; fits 64 GB only at 4-bit (~40 GB), tight |
| Muse Glimmer 30B | `meta-models/Muse-Glimmer-30B` | ~60 GB | Apache 2.0, distilled from Muse Spark for local agents |
| Muse Spark 1.x | — | — | **Not downloadable** — closed weights, API-only; use Glimmer locally |
| R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | ~64 GB | Reasoning traces; weak at tool calling — benchmark, don't agent |
| gpt-oss-20b | `openai/gpt-oss-20b` | ~40 GB | Apache 2.0, built for agents/tool use |

</details>

## 3. Shortcut — pre-quantized MLX ports (skip steps 2–3)

Download a ready-made 4-bit build and jump to README step 4. You learn less,
but you gain a second data point: *their* quant vs *yours*.

- [ ] Port downloaded to `models/<name>-mlx-4bit`
- [ ] Benched with `bench.py --model …` and compared against your own quant

| Port | HF ID |
|---|---|
| Qwen3-32B 4-bit | `mlx-community/Qwen3-32B-4bit` |
| Qwen3.5-4B 4-bit | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| Qwen3.5-9B 4-bit | `mlx-community/Qwen3.5-9B-MLX-4bit` |
| Qwen3.5-35B-A3B 4-bit | `mlx-community/Qwen3.5-35B-A3B-4bit` |
| Qwen3.6-35B-A3B 4-bit | `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` |
| Qwen3.8-27B 4-bit | `mlx-community/Qwen3.8-27B-4bit` |
| Qwen3-Coder-30B 4-bit | `lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit` |
| Qwen2.5-Coder-32B 4-bit | `lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-4bit` |
| Llama-3.3-70B 4-bit | `mlx-community/Llama-3.3-70B-Instruct-4bit` |
| Muse-Glimmer-30B 4-bit | `mlx-community/Muse-Glimmer-30B-4bit` |

```bash
uv run huggingface-cli download <PORT_ID> --local-dir "models/<name>-mlx-4bit"
uv run python bench.py --model "models/<name>-mlx-4bit"
```

`mlx-community` and `lmstudio-community` are the two publishers the
[MLX model explorer](https://huggingface.co/spaces/mlx-community/mlx-model-explorer)
indexes — when both offer a port, try both and keep the faster.

## 4. Where model locations are defined

Everything is an env var or a flag — no hardcoded paths:

| Knob | Default | Used by |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-8B` | `run_all.sh` (all three dirs derive from it) |
| `models/<short>-hf` | from `MODEL_ID` | HF download target (step 1) |
| `models/<short>-mlx-bf16` | from `MODEL_ID` | convert output (step 2) → quantize input (step 3) |
| `models/<short>-mlx-4bit` | from `MODEL_ID` | quantize output (step 3); default bench target |
| `bench.py --model` | bf16 + 4-bit pair | Bench one dir instead of the pair |
| `bench_engines.py --base-url --model --tag` | (required) | Which server and served-model name to race |
| Rapid-MLX alias | `qwen3.5-9b-4bit` | README step 6 (`rapid-mlx serve <alias>`) |

Try it:

```bash
MODEL_ID=Qwen/Qwen3-30B-A3B ./run_all.sh   # whole pipeline on a bigger MoE
uv run python bench.py --model models/Qwen3-30B-A3B-mlx-4bit
```

## Worked example — Qwen3.5-35B-A3B end to end

- [ ] ~160 GB free (`df -h`) — all three copies coexist mid-pipeline
- [ ] Full run finished and `REPORT.md` read
- [ ] Served, raced, and compared against the community port

```bash
export HF_TOKEN="hf_..."
export MODEL_ID=Qwen/Qwen3.5-35B-A3B
./run_all.sh   # download (~70 GB) -> bf16 -> 4-bit -> infer -> bench -> REPORT.md
```

Serve your build and race it:

```bash
# terminal 1
uv run mlx_lm.server --model models/Qwen3.5-35B-A3B-mlx-4bit --port 8000
# terminal 2
uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3.5-35B-A3B-mlx-4bit --tag mlx-lm
```

Or A/B your quant against the community's (~20 GB, straight to step 4):

```bash
uv run huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit \
  --local-dir models/Qwen3.5-35B-A3B-mlx-4bit-community
uv run python bench.py --model models/Qwen3.5-35B-A3B-mlx-4bit-community
```

## Worked example — Gemma 4 (26B-A4B MoE and 31B dense)

Gemma 4 is Google's 2026 family and `mlx-lm` supports it natively (`gemma4`
architecture). The MoE is the interesting one: 26B total parameters with only
~4B active per token — near-27B quality at small-model speed and ~15 GB in
memory at 4-bit.

- [ ] License accepted on the model's HF page if the download asks (Google
      models sometimes gate files even when metadata is public)
- [ ] Full run finished and `REPORT.md` read

```bash
export HF_TOKEN="hf_..."
export MODEL_ID=google/gemma-4-26B-A4B-it
./run_all.sh   # download (~52 GB) -> bf16 -> 4-bit -> infer -> bench -> REPORT.md
```

Same commands, dense alternative (~62 GB download, ~18 GB at 4-bit):

```bash
export MODEL_ID=google/gemma-4-31B-it
./run_all.sh
```

Serve your build and race it (same pattern as every model — only names change):

```bash
# terminal 1
uv run mlx_lm.server --model models/gemma-4-26B-A4B-it-mlx-4bit --port 8000
# terminal 2
uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \
  --model gemma-4-26B-A4B-it-mlx-4bit --tag mlx-lm
```

Suggested A/B once both are built: Gemma 4 26B-A4B vs Qwen3.5-35B-A3B —
2026 MoE vs 2026 MoE, same prompts, `bench.py --model` on each 4-bit dir,
then `make_report.py`. That table is the one worth committing.

## 5. Runner options — four ways to serve a model

<details open>
<summary><strong>A. mlx-lm (this lab's path) — maximum learning</strong></summary>

Convert HF → MLX yourself, serve with `mlx_lm.server`, benchmark in-process.
Apple-native speed. README steps 1–5.

</details>

<details>
<summary><strong>B. Ollama (easiest) — results, not lessons</strong></summary>

One tool for download + serve + API. Recent Ollama routes supported models
through an MLX backend on Apple Silicon; weights are GGUF, not MLX-native.

```bash
brew install ollama
ollama pull gpt-oss:20b     # library names: qwen3, qwen3.5, llama3.3, gemma3, deepseek-r1…
ollama run gpt-oss:20b
```

Race it: `ollama serve`, then
`bench_engines.py --base-url http://localhost:11434/v1` — compare against
your MLX-native run of the same-class model.

</details>

<details>
<summary><strong>C. llama.cpp (maximum control) — isolate the layer</strong></summary>

The engine under Ollama's hood, without the wrapper. GGUF weights + explicit
flags for threads, batch, cache, quant:

```bash
brew install llama.cpp
uv run huggingface-cli download ggml-org/gpt-oss-20b-GGUF \
  --include "*.Q4_K_M.gguf" --local-dir models/gguf
llama-server -m models/gguf/*.gguf -c 8192 --port 8001
```

Then `bench_engines.py --base-url http://127.0.0.1:8001/v1 --tag llama-cpp`.
Use this when a number looks wrong and you want to know which layer (app,
server, engine) owns it.

</details>

<details>
<summary><strong>D. Rapid-MLX (the claim to test)</strong></summary>

MLX-native server, OpenAI-compatible, claims up to 4× Apple's mlx-lm on the
same weights:

```bash
brew install rapid-mlx
rapid-mlx serve qwen3.5-9b-4bit          # localhost:8000, OpenAI-compatible
rapid-mlx benchmark run qwen3.5-9b-4bit  # fixed protocol — NOT comparable to serving
```

Race it with `bench_engines.py --tag rapid-mlx` per README step 6.

</details>

Rule of thumb: learn with A, daily-drive with B, debug with C, verify hype
with D.
