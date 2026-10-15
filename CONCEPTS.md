# Concepts behind the lab

Every step of the pipeline, and every jargon term it uses, in one place.
Format per concept: **what** it is, **why** it matters here, and **where**
you meet it in this repo.

## How to use this guide

| You have… | Read… |
|---|---|
| 5 minutes before running anything | [The big picture](#the-big-picture) + [Each pipeline step, in one line](#each-pipeline-step-in-one-line) |
| 20 minutes alongside the lab | Each section's main text; skip the collapsed "Going deeper" boxes |
| An evening and curiosity | Everything, including the collapsed boxes and the ✋ exercises |

> ✋ = pause-and-try exercises. They are the actual learning — do them.

## The big picture

An **LLM** (large language model) is a next-token predictor: given a sequence
of **tokens**, it outputs a probability distribution over what comes next.
Sampling that distribution repeatedly produces text. Everything below is
machinery for storing that predictor, feeding it input, and measuring how
fast it predicts.

**Inference** is simply running a trained model to get outputs (as opposed to
*training*, which adjusts its weights). All of this repo is inference: nobody
here trains anything.

> ✋ Try it: open `results/inference_*.json` after README step 4 and find one
> answer where 4-bit visibly differs from bf16. That gap is the entire
> quantization tradeoff in one example.

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

> ✋ Try it: after step 3, run `du -sh models/*` and compute your actual
> bf16 → 4-bit ratio. Then predict: if decode speed ≈ bandwidth ÷ model
> bytes, how much faster *should* 4-bit decode be? Write it down — step 5
> grades your prediction.

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

<details>
<summary>Going deeper: why temperature 0 matters for benchmarks</summary>

A benchmark is a measurement instrument, and sampling randomness is noise in
the instrument. At temperature 0.7, two runs of the same prompt take different
token paths — different lengths, different early-EOS behavior — so tok/s
jitters for reasons that have nothing to do with the engine. Temperature 0
fixes the token path, leaving only system noise. That is also why
`bench_engines.py` sets it explicitly instead of trusting server defaults.

</details>

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

> ✋ Try it: run one prompt through `mlx_lm.generate` and watch it stream.
> The pause before the first word is TTFT (prefill); the typing speed after
> is decode. You now know which number each sensation maps to.

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

<details>
<summary>Going deeper: the O(n²) → O(n) trade, concretely</summary>

Without a cache, generating token *t* re-attends over all *t−1* previous
tokens, so a 1000-token reply costs ~1000² attention operations. With the KV
cache, each step attends over cached entries in one pass — but the cache
itself costs `layers × 2 (K and V) × tokens × head_dim × bytes` of memory.
That is why long contexts are a *memory* problem first and a speed problem
second, and why halving precision (4-bit) only halves the weights, never the
full story.

</details>

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

> ✋ Try it: comment out the warmup line in `bench.py`, run twice, and watch
> the first prompt's numbers crater. Restore it. You will never again trust a
> benchmark that doesn't mention warmup.

## Each pipeline step, in one line

- [ ] I can explain what each step changes and why it exists

| Step | What | Why |
|---|---|---|
| 1. Download | Fetch safetensors + tokenizer + config from HF | You need the raw artifact; seeing its layout demystifies "a model" |
| 2. Convert (bf16) | Repackage weights as MLX `.npy` shards | Apple's runner needs Apple's layout; isolates format change from precision change |
| 3. Quantize (4-bit) | Shrink weights to ~¼ size | Fit + speed; the fundamental local-LLM tradeoff |
| 4. Inference test | Read actual answers, bf16 vs 4-bit | Speed is meaningless if quality broke — check quality first |
| 5. Benchmark | TTFT, prefill/decode tok/s, peak memory | Numbers replace vibes; warmup + repeats replace luck |
| 6. Engine race | Same prompts vs mlx-lm and Rapid-MLX servers | Test vendor claims on your hardware, your weights |
| 7. Report | Merge JSON into REPORT.md | A benchmark you can't re-read is a benchmark you didn't run |

## Hardware factors: what actually limits you

Your benchmark numbers are the product of a whole stack. When a number
surprises you, the cause is almost always one of these. Read the bold lines
first; expand a box only when that component is your suspect.

**GPU (the workhorse).** Does the parallel math — the big matrix multiplies
of prefill and attention. More cores help prefill and training; decode (one
token at a time) barely uses them, which is why a "faster GPU" alone often
disappoints on chat speed. Your M5 Max GPU also carries per-core Neural
Accelerators (added in the M5 generation) that speed specific AI ops.

**CPU (the orchestrator).** Runs the Python loop, tokenization, sampling,
networking, and the server itself. A slow or oversubscribed CPU shows up as
jittery inter-token latency, not low average tok/s. Rule of thumb: if tok/s
is fine but streaming stutters, suspect the CPU side.

**Memory capacity (the ceiling).** Weights + KV cache + framework overhead must
fit in unified memory or nothing runs. Budget it explicitly:
weights (params × bytes-per-param) + KV cache (grows with context × layers) +
~2–4 GB for macOS, the Python process, and Metal buffers. On your 64 GB Mac
that means roughly: 30B models comfortable at 4-bit, 70B possible but tight,
and long-context sessions can OOM from cache growth while weights fit fine.

**Memory bandwidth (the speed limit).** Decode reads every weight once per
token, so decode tok/s ≈ bandwidth ÷ model-bytes. This is why 4-bit is ~2–4×
faster than bf16 and why Apple Silicon (hundreds of GB/s) beats machines with
bigger GPUs but narrower buses. Check your chip's published bandwidth, then
divide: predicted tok/s ≈ bandwidth ÷ weight-size is usually within 30% of
measured — a great sanity check on your benchmark.

<details>
<summary>Going deeper: caches, storage, overhead, OS limits</summary>

**Caches (the hidden accelerators).** Three layers, smallest-first: GPU
registers/shared memory (per-operation, handled by kernels), system-level
cache on the SoC (shared by CPU/GPU, why unified memory is efficient), and
the software KV cache (above). You don't manage the first two; you observe
them as "why is the second run of the same prompt faster" alongside warmup.

**Storage (the waiting room).** SSD speed only matters at load time and when
memory-mapping huge models — never tok/s once running. What matters is
*capacity*: keep 2–3× the model size free. NVMe on modern Macs loads a 16 GB
model in seconds; verify with `time` around step 1 vs step 5.

**Overhead (the tax).** Python dispatch per token, Metal kernel-launch
latency, HTTP/SSE framing, tokenizer encode/decode, chat-template
formatting. Collectively they floor TTFT and cap small-model speed (a 1B
model won't hit theoretical tok/s — overhead dominates, not bandwidth).
In-process `bench.py` vs `bench_engines.py` on the same weights shows you the
server overhead directly.

**OS factors (macOS).** Memory pressure past ~85–90% triggers compression and
swapping — cliff-edges, so leave headroom. Background apps steal bandwidth,
not just capacity — close them for clean numbers. Thermal throttling
downclocks sustained runs on fanless machines; if late repeats are slower
than early ones, suspect heat. Never benchmark in Low Power Mode.

**Other limitations.** Context length trades against speed and memory (KV
cache grows per token). Batch size 1 (this lab) is worst-case throughput;
servers gain by batching. Benchmark plugged in. And reproducibility: close
apps, fix temperature at 0 for races, report chip/RAM/runs with every number
— a tok/s without its conditions is trivia.

</details>

> ✋ Try it (the capstone): predict your 4-bit decode tok/s from bandwidth ÷
> weight size before running `bench.py`, then compare. The gap between
> prediction and measurement is the sum of every overhead above — now you
> know its name.

## Further reading (in order)

1. Your own `REPORT.md` after a full run — your numbers beat any article.
2. `rapid-mlx benchmark plan <alias>` — see a vendor's fixed protocol, then
   compare it against your live-serving numbers and notice the gap.
3. The mlx-lm [SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
   — what flags your `mlx_lm.server` step actually supports.
