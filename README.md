# MLX benchmark lab

Learn the full local-LLM loop on Apple Silicon, then verify vendor speed
claims yourself: convert and quantize a Hugging Face model to MLX, benchmark
Apple's `mlx-lm`, and race it against
[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) over an OpenAI-compatible
endpoint. Same prompts, same machine, your own numbers.

New to the terminology (quantization, KV cache, TTFT, …)? Read
[CONCEPTS.md](CONCEPTS.md) first — it explains every step's why/what/how.
Models, download IDs, and Ollama / llama.cpp alternatives live in
[MODELS.md](MODELS.md). Who tests what, why, and how to read the numbers:
[TESTING.md](TESTING.md).

## Choose your path

| Path | Time | You will run | Start |
|---|---|---|---|
| **A. Full lab** (learn everything) | ~1–2 hrs | Steps 0–7 below, in order | Step 0 |
| **B. Fast lane** (benchmark today) | ~20 min | Steps 0–1 (shortcut), 4–5, 7 | Step 0, then the shortcut in step 1 |
| **C. Engine race only** (test the hype) | ~30 min | Step 6 | You need any served model; step 1 shortcut is fastest |

Default model is small on purpose (`Qwen/Qwen3-8B`, ~16 GB download) so the
loop finishes in one sitting. Scale up with `MODEL_ID` once you trust it.

## Step 0. Setup (everyone, once)

- [ ] Apple Silicon Mac, Python 3.12+, `uv`, `git`, `curl`
- [ ] ~40 GB free disk (`df -h /`) for the 8B default
- [ ] A [Hugging Face token](https://huggingface.co/settings/tokens) — **required**
      (how to create one: [MODELS.md](MODELS.md#1-catalog--source-weights-by-vendor))

```bash
uv venv --python 3.12
uv sync
export HF_TOKEN="hf_..."   # required: every download step uses it
```

> `uv run` below uses the project-local `.venv` automatically — no activation needed.

## The pipeline

Tick each box as you go. Every step shows the command, what to check, and
where to learn more. Scripts discover local builds automatically — `ls models/`
or any script's `--list` shows what a default run covers, and every script ends
with a summary (Ctrl+C keeps partial results instead of a traceback).

### 1. Download the weights ⬇️

- [ ] Weights downloaded to `models/`

```bash
uv run hf download "${MODEL_ID:-Qwen/Qwen3-8B}" --local-dir "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-hf"
# default MODEL_ID=Qwen/Qwen3-8B — try MODEL_ID=Qwen/Qwen3-30B-A3B for bigger
```

<details>
<summary>What just happened? (click to expand)</summary>

A model repo is a versioned folder of safetensors + tokenizer + config.
Run `du -sh models/*` — that number is the bf16 truth everything below
shrinks. Other models, gated-model login, and pre-quantized shortcuts:
[MODELS.md](MODELS.md). Concepts: [CONCEPTS.md](CONCEPTS.md#weights-formats-and-quantization).

**Shortcut (Path B):** skip steps 2–3 — download a ready-made 4-bit build
and jump to step 4:

```bash
uv run hf download mlx-community/Qwen3-8B-4bit --local-dir models/Qwen3-8B-mlx-4bit
```

</details>

### 2. Convert to MLX 🔄

- [ ] bf16 MLX copy exists and is ~same size as the download

```bash
uv run mlx_lm.convert \
  --hf-path "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-hf" \
  --mlx-path "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-mlx-bf16"
```

<details>
<summary>What just happened?</summary>

Same information, new layout: MLX `.npy` shards + `config.json`. No precision
was lost — this step isolates the *format* change from the *precision* change
in step 3. Compare: `du -sh models/*-hf models/*-bf16`.

</details>

### 3. Quantize to 4-bit 🗜️

- [ ] 4-bit copy is ~¼ the size of bf16

```bash
uv run mlx_lm.convert \
  --hf-path "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-mlx-bf16" \
  --mlx-path "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-mlx-4bit" \
  -q --q-bits 4
```

<details>
<summary>What just happened?</summary>

Weights re-stored at 4-bit: ~4× smaller, ~2–4× faster decode (decode speed ≈
memory bandwidth ÷ model bytes). The price is rounding error — which is why
step 4 comes *before* you trust any speed number.
Background: [CONCEPTS.md](CONCEPTS.md#weights-formats-and-quantization).

</details>

### 4. Smoke-test inference 💬

- [ ] One-liner answers coherently
- [ ] Full transcripts saved and skimmed (`results/inference_*.json`)

```bash
uv run mlx_lm.generate --model "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-mlx-4bit" \
  --prompt "Explain quantization in one paragraph." --max-tokens 150
uv run python inference_test.py   # every prompt in prompts.txt, bf16 vs 4-bit
```

<details>
<summary>What am I looking for?</summary>

Your qualitative gate: does 4-bit still answer coherently? If bf16 is sharp
and 4-bit is mush, no tok/s number can save it. Read the transcripts before
step 5.

</details>

### 5. Benchmark mlx-lm 📊

- [ ] `results/bench_*.json` written for bf16 and 4-bit

```bash
uv run python bench.py
```

<details>
<summary>What do the numbers mean?</summary>

Per prompt, after a discarded warmup run: **TTFT** (snappiness),
**prefill tok/s** (prompt processing), **decode tok/s** (the headline —
generation dominates chat time), and **peak memory**. Decode ≫ prefill in
importance. Full decoder ring:
[CONCEPTS.md](CONCEPTS.md#speed-latency-ttft-and-throughput).

**Predict first:** look up your chip's memory bandwidth, divide by the 4-bit
model size — that's your theoretical decode tok/s. The gap to measured is
overhead, and now you know its name.

</details>

### 6. Race engines 🏁 (mlx-lm vs Rapid-MLX vs Ollama)

- [ ] `results/engines_mlx-lm.json` and `results/engines_rapid-mlx.json` written
- [ ] `results/engines_ollama-*.json` written (if you have Ollama models)

Terminal 1 — serve your 4-bit build with Apple's server:

```bash
uv run mlx_lm.server --model "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-mlx-4bit" --port 8000
```

Terminal 2 — benchmark it (same prompts, temperature 0 for identical work):

```bash
uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \
  --model "models/$(basename ${MODEL_ID:-Qwen/Qwen3-8B})-mlx-4bit" --tag mlx-lm
# ^ must match the server's --model string EXACTLY (full path, not basename):
# mlx_lm.server routes by request model name and 404s anything else.
```

Terminal 3 — serve Rapid-MLX (install once: `brew install rapid-mlx`).
`serve` blocks, so run the bench command in a *separate* terminal:

```bash
rapid-mlx serve qwen3.5-9b-4bit
```

```bash
# ⚠️ if port 8000 is taken (mlx_lm.server still up), rapid-mlx reports
# "using 8001 instead" — match --base-url to the port it actually uses.
uv run python bench_engines.py --base-url http://localhost:8001/v1 \
  --model "qwen3.5-9b-4bit" --tag rapid-mlx
# Pareto sweep (throughput-vs-latency frontier across load levels):
uv run python bench_engines.py --base-url http://localhost:8001/v1 \
  --model "qwen3.5-9b-4bit" --tag rapid-mlx-sweep --repeats 1 --sweep 1,2,4
```

Console prints an aiperf-style table per race (TTFT / ITL / request latency /
token throughput with avg·min·max·p99·p90·p50·std); `--concurrency N` overlaps
requests to measure real contention.

Terminal 4 — race llama.cpp (bare engine, explicit knobs):

```bash
llama-server -m models/gguf/Qwen3-8B-Q4_K_M.gguf -c 4096 --port 8001
uv run python bench_llamacpp.py --base-url http://127.0.0.1:8001/v1 \
  --model Qwen3-8B-Q4_K_M.gguf
# discover: uv run python bench_llamacpp.py --list
```

Terminal 5 — race Rapid-MLX by alias:

```bash
rapid-mlx serve qwen3.5-9b-4bit   # :8001 if :8000 is taken
uv run python bench_rapid.py --alias qwen3.5-9b-4bit --port 8012
```

Terminal 6 — race your already-downloaded Ollama models (no setup beyond
`ollama serve`, which the desktop app also provides):

```bash
uv run python bench_ollama.py --repeats 1   # races everything in `ollama list`
# subset: uv run python bench_ollama.py --models qwen3.6:27b-mlx,gemma4:12b-mlx
```

Discovers models via Ollama's API (runner, size, quant recorded per model),
skips non-chat models (embed/decision-only) instead of failing, and writes one
`results/engines_ollama-<model>.json` per model — picked up automatically by
step 7's race table.

<details>
<summary>How to read the result (methodology)</summary>

Same-class models, not identical weights — Rapid-MLX serves its own aliases,
so identical-weights A/B is only possible if it accepts a local path (check
`rapid-mlx serve --help`). Same prompts, same machine, streaming TTFT +
decode rate: still a fair server-vs-server race. Rapid-MLX's own
`rapid-mlx benchmark run <alias>` uses a fixed non-speculative protocol that
is *not* comparable to live-serving numbers, including ours.

</details>

### 7. Generate the report 📝

- [ ] All six reports written and read

```bash
uv run python make_report.py
# writes reports/REPORT.md + .html + .pdf + .xlsx + .json + .csv
# single format: uv run python make_report.py --format pdf
```

Merges every `results/*.json` into six formats: machine specs, on-disk
sizes, bf16-vs-4bit deltas, and the engine race table — as Markdown, styled
HTML, PDF, spreadsheet (summary + per-run sheets with filters), raw JSON,
and flat CSV.

### Rerun everything

```bash
./run_all.sh   # steps 1–5 + 7 (engines in step 6 stay interactive)
```

## Results

Results are yours and gitignored (`results/*.json`, `models/`). Commit
`reports/*` only when you want to publish a run.

## Scripts (what each file is for)

| File | Step | Does |
|---|---|---|
| `run_all.sh` | 1–5, 7 | Whole mlx-lm pipeline; `MODEL_ID=<id>` overrides the model |
| `bench.py` | 5 | In-process TTFT/prefill/decode/memory; defaults to all local builds |
| `inference_test.py` | 4 | Quality gate transcripts, bf16 vs 4-bit |
| `bench_engines.py` | 6 | Server-vs-server race over OpenAI SSE; shared by Ollama racer |
| `bench_ollama.py` | 6 | Races everything in `ollama list`; skips non-chat models |
| `bench_llamacpp.py` | 6 | GGUF discovery (`--list`) + llama-server race, tagged `llama-cpp` |
| `bench_rapid.py` | 6 | Rapid-MLX alias race (+ `--sweep`), tagged `rapid-mlx` |
| `import_aiperf.py` | 6 | Imports aiperf JSON exports as labeled aggregate rows |
| `make_report.py` | 7 | Merges JSON into md/html/pdf/xlsx/json/csv (`--format`) |
| `labutil.py` | all | Completion summaries + graceful Ctrl+C (exit 130, keeps partials) |
| `prompts.txt` | 4–6 | The shared prompt set — edit to race your own workload |

`--list` on any bench/test script prints what a default run would cover.

## Troubleshooting

<details>
<summary>Connection refused on <code>bench_engines.py</code></summary>

No server is listening. Start one in a separate terminal first
(`uv run mlx_lm.server --model models/Qwen3-8B-mlx-4bit --port 8000` for
mlx-lm; `rapid-mlx serve <alias>`; `ollama serve`), then re-run. The script
says this instead of a traceback — believe it.

</details>

<details>
<summary>Server 404 on <code>/v1/chat/completions</code></summary>

`mlx_lm.server` routes by request model name: `--model` in the bench command
must equal the server's `--model` string exactly (full path, not basename).
Anything else makes the server look the name up on HF and 404.

</details>

<details>
<summary><code>HF_TOKEN</code> / 401 / 403 errors on download</summary>

Export the token (`export HF_TOKEN="hf_..."`) and log in:
`uv run hf auth login --token "$HF_TOKEN"`. Gated models (Llama,
Gemma, Qwen-Coder) also need the license accepted on their HF page. Details:
[MODELS.md](MODELS.md).

</details>

<details>
<summary>Out of disk or memory</summary>

All three copies (HF + bf16 + 4-bit) coexist mid-pipeline — 3× the bf16 size
free is the rule. OOM with weights that "should fit" is usually KV-cache
growth: shorten context / `max-tokens`. Background:
[CONCEPTS.md](CONCEPTS.md#hardware-factors-what-actually-limits-you).

</details>

<details>
<summary>Numbers look wrong</summary>

Close browsers and Electron apps (they steal memory bandwidth), stay plugged
in, never benchmark in Low Power Mode, and confirm one warmup run was
discarded. Still off? Reproduce via `llama.cpp` to isolate the layer —
[MODELS.md §4](MODELS.md#4-runner-options--four-ways-to-serve-a-model).

</details>

<details>
<summary>Rapid-MLX 400: <code>function.name must be a non-empty string</code></summary>

That error is not the benchmark — our scripts send no tools. It comes from an
agent client (Claude Code, OpenClaw, etc.) whose request carries 100+ tool
definitions, one with an empty or illegal name. Rapid-MLX validates tool names
against the OpenAI spec (`^[a-zA-Z0-9_-]{1,64}$`) and rejects the whole
request. Find the offender in the *client*: look for a dynamically generated
or MCP tool with an empty/missing name, rename it, and retry. The server log
line `tools=N` tells you which request failed.

</details>

## FAQ

- **Ollama vs MLX?** Recent Ollama routes some models through an MLX backend,
  but its files are GGUF. Here you work with MLX-native weights — no
  translation layer.
- **Why is decode tok/s the headline?** Prefill is one parallel pass; decode
  emits one token at a time and dominates chat wall-clock.
- **Why warmup?** First run pays kernel-compilation and cache costs. We
  discard it so numbers reflect steady state.

## Contributors

- [Nael Mohammad](https://github.com/naelmohammad) — author and maintainer

## License

MIT — see [LICENSE](LICENSE).
