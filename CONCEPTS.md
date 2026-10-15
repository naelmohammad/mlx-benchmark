# Concepts behind the lab

Every step of the pipeline, and every jargon term it uses, in one place.
Format per concept: **what** it is, **why** it matters here, and **where**
you meet it in this repo.

## The big picture

An **LLM** (large language model) is a next-token predictor: given a sequence
of **tokens**, it outputs a probability distribution over what comes next.
Sampling that distribution repeatedly produces text. Everything below is
machinery for storing that predictor, feeding it input, and measuring how
fast it predicts.

**Inference** is simply running a trained model to get outputs (as opposed to
*training*, which adjusts its weights). All of this repo is inference: nobody
here trains anything.

## The hardware: Apple Silicon and unified memory

**Apple Silicon** (M1/M2/…/M5) puts CPU, GPU, and Neural Engine on one chip
sharing one pool of RAM — **unified memory**. A Mac with 64 GB RAM gives the
GPU ~64 GB of usable model space, where a typical desktop GPU has 8–24 GB of
VRAM. That is the entire reason local LLMs on Mac work: a 30–70B-parameter
model can sit in memory with no copying between CPU and GPU.

*Where:* `make_report.py` records your chip and RAM; the benchmark numbers
only make sense relative to that hardware.

## The framework: MLX

**MLX** is Apple's array framework for Apple Silicon (think "NumPy that runs
on the GPU, with lazy evaluation"). `mlx-lm` is the Hugging Face–style
toolkit on top of it: `convert` (reformat weights), `generate` (run text),
`server` (serve an API). It exists because generic runners (built for NVIDIA)
leave Apple-Silicon performance on the table.

**Ollama**, by contrast, is a model *runner/distribution tool*: one download,
one command, chat UI and API included. Historically it ran GGUF weights via
llama.cpp; recent versions route some models through an MLX backend on Macs.
**LM Studio** is a similar one-click runner with MLX support. This lab skips
the friendly runners on purpose: converting weights yourself teaches what
they hide.

## Weights, formats, and quantization

Model **weights** are the billions of learned numbers. Downloaded at full
**bf16** precision (2 bytes each), an 8B model is ~16 GB.

**Quantization** re-stores those numbers at lower precision — here **4-bit**
(~0.5 bytes each, ~4× smaller). Why: LLM inference is *memory-bandwidth
bound* (see below), so moving 4× less data per token makes generation roughly
faster and lets bigger models fit. The cost: rounding error, occasionally
visible as worse answers — which is exactly why step 4 (inference test)
exists *before* you trust step 5's speed numbers. **Group size** (default 64)
controls the granularity: each block of 64 weights shares a scale factor.
Smaller groups = more accurate, slightly larger.

*Where:* steps 2–3 (`mlx_lm.convert`, then `convert -q --q-bits 4`).
Compare `du -sh` of the bf16 vs 4-bit dirs — that ratio is quantization made
visible. Related formats you'll meet elsewhere: **GGUF** (llama.cpp/Ollama's
container, same idea, different layout) and **AWQ/GPTQ** (older GPU-targeted
quant formats).

## Tokens, context, and chat templates

A **token** is the model's unit of text — roughly ¾ of an English word.
"Quantization" might be `Quant` + `ization`. The **tokenizer** (shipped with
the model) maps text ↔ token IDs; different models tokenize differently, so
tok/s numbers are only comparable *within* one model family.

**Prompt context** (context window) is how many tokens the model can attend
to at once — input plus output combined. Exceed it and the start is forgotten
or the run errors. Small local models typically offer 32K–256K tokens.

A **chat template** wraps your raw prompt in the control tokens the model was
trained with (`<|user|>…<|assistant|>` etc.). Skip it and the model sees an
alien format and answers worse. The scripts apply it automatically when the
tokenizer provides one.

**Temperature** (0.0–2.0) controls sampling randomness: 0 = always the most
likely token (deterministic — used in the engine race so both servers do
identical work), ~0.7 = natural chat variety.

*Where:* `prompts.txt`, `inference_test.py` (`TEMPERATURE`), `bench_engines.py`
(`temperature: 0.0`, with the reason in a comment).

## Speed: latency, TTFT, and throughput

**Latency** is "how long until I get X". For chat you care about two of them:

- **TTFT (time-to-first-token):** how long from sending the prompt until the
  first token streams back. This is what the user *feels* — the difference
  between an app that responds instantly and one that hangs.
- **Inter-token latency / TPOT (time-per-output-token):** the gap between
  consecutive tokens — what makes text feel like it types fast or slow.

**Throughput (tok/s)** is tokens divided by time. Prefill tok/s and decode
tok/s are different operations (next section) — never average them.

Why decode dominates: a chat reply spends ~5% of its time on the prompt and
~95% generating, one token at a time. Decode tok/s is therefore the headline
number for interactive use; TTFT is the headline for perceived snappiness.

*Where:* `bench.py` reports prompt_tps (prefill) and generation_tps (decode)
separately; `bench_engines.py` measures TTFT over streamed SSE.

## Prefill, decode, and the KV cache

Generation has two phases. **Prefill** processes the whole prompt in one
parallel pass (fast, compute-heavy). **Decode** then emits one token at a
time, each step re-reading the model weights *and everything generated so
far*.

Recomputing attention over all previous tokens every step would be quadratic
— so the model caches them. The **KV cache** stores the Key and Value
projections of every processed token, per layer. Each new token attends to
the cache instead of recomputing it: O(n) memory per step instead of O(n²)
compute. The cache lives in unified memory and grows with context length —
long conversations on small VRAM machines die from KV-cache growth, not from
weights.

**Prompt caching** is the server-level cousin: if a new request shares a
prefix with a previous one (system prompt, document), the server reuses the
saved KV entries and skips prefill for that prefix — lower TTFT, lower cost.
Same cache, one level up the stack.

*When it matters:* multi-turn chats, agents re-sending conversation history,
and RAG systems reusing long retrieved contexts. Watch peak memory in
`bench.py` grow as `max_tokens` grows — that delta is mostly KV cache.

## Memory-bandwidth bound, warmup, and MTP

LLM decode is **memory-bandwidth bound**: generating one token means reading
*all* weights from memory to do relatively little math, so speed ≈ how fast
memory feeds the chip — not how fast the chip multiplies. Consequences:

1. Quantization speeds up decode (fewer bytes to move per token).
2. Bigger chips help less than faster memory; your M5 Max's bandwidth is the
   asset, not just its cores.
3. Batching (serving many users at once) amortizes one weight-read across
   many tokens — why servers beat single-user scripts on throughput.

**Warmup:** the first run compiles GPU kernels and fills caches, so it is
always slower. Every script here runs one untimed warmup pass and discards it
— without that, you benchmark the compiler, not the model.

**MTP (multi-token prediction) / speculative decoding:** instead of one token
per weight-read, a small draft model (or extra heads) guesses several tokens
and the big model verifies them in one pass. Correct guesses = multiple
tokens per read = higher tok/s. Rapid-MLX's speed claims partly rest on this;
that is also why their fixed-protocol `benchmark run` numbers and live
serving numbers are different things (README step 6).

## Each pipeline step, in one line

| Step | What | Why |
|---|---|---|
| 1. Download | Fetch safetensors + tokenizer + config from HF | You need the raw artifact; seeing its layout demystifies "a model" |
| 2. Convert (bf16) | Repackage weights as MLX `.npy` shards | Apple's runner needs Apple's layout; isolates format change from precision change |
| 3. Quantize (4-bit) | Shrink weights to ~¼ size | Fit + speed; the fundamental local-LLM tradeoff |
| 4. Inference test | Read actual answers, bf16 vs 4-bit | Speed is meaningless if quality broke — check quality first |
| 5. Benchmark | TTFT, prefill/decode tok/s, peak memory | Numbers replace vibes; warmup + repeats replace luck |
| 6. Engine race | Same prompts vs mlx-lm and Rapid-MLX servers | Test vendor claims on your hardware, your weights |
| 7. Report | Merge JSON into REPORT.md | A benchmark you can't re-read is a benchmark you didn't run |

## Further reading (in order)

1. Your own `REPORT.md` after a full run — your numbers beat any article.
2. `rapid-mlx benchmark plan <alias>` — see a vendor's fixed protocol, then
   compare it against your live-serving numbers and notice the gap.
3. The mlx-lm [SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
   — what flags your `mlx_lm.server` step actually supports.
