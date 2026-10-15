# Testing & reporting guide

Who this is for, what gets tested, why each test exists, how to run it,
where results go, and when to run what.

## Who

- **You, learning local LLMs** — run the full loop once; every step teaches
  one concept (see [CONCEPTS.md](CONCEPTS.md)).
- **You, comparing models/engines** — skip to the race; the report tables do
  the arguing.
- **You, publishing numbers** — follow the reproducibility rules at the bottom
  so others can trust your `reports/`.

## What is tested (and why)

| Test | Script | What it answers | Why it exists |
|---|---|---|---|
| Inference transcripts | `inference_test.py` | Did 4-bit break coherence? | Speed is meaningless if quality broke — quality gate runs *first* |
| In-process benchmark | `bench.py` | prefill tok/s, decode tok/s, TTFT proxy, peak memory per build | Baseline numbers with no server in the loop |
| Server race | `bench_engines.py` | TTFT + decode rate over OpenAI SSE, per engine | Compares servers (mlx-lm vs Rapid-MLX vs anything) fairly |
| Ollama race | `bench_ollama.py` | Same, for everything in `ollama list` | Uses models you already downloaded; skips non-chat ones |
| Report build | `make_report.py` | Merges all JSON into md/html/pdf/xlsx/json/csv + charts | A benchmark you can't re-read is one you didn't run |

## How — commands

```bash
# - [ ] discover what a default run covers (no model hardcoded anywhere)
uv run python bench.py --list
uv run python inference_test.py --list

# - [ ] quality gate first (read the transcripts!)
uv run python inference_test.py

# - [ ] baseline numbers (bf16 + 4-bit, or --model <one dir>)
uv run python bench.py

# - [ ] server race (server running in ANOTHER terminal first)
uv run mlx_lm.server --model models/Qwen3-8B-mlx-4bit --port 8000
uv run python bench_engines.py --base-url http://127.0.0.1:8000/v1 \
  --model "models/Qwen3-8B-mlx-4bit" --tag mlx-lm

# - [ ] everything already in Ollama, one command
uv run python bench_ollama.py --repeats 1

# - [ ] all six report formats + charts
uv run python make_report.py
```

## Where

- **Inputs:** `prompts.txt` (the shared workload — edit it to test *your*
  workload), `models/` (auto-discovered; never hardcoded).
- **Raw results:** `results/*.json` — gitignored, per-run data with nulls
  where a metric was unavailable, `error` fields where a run failed.
- **Reports:** `reports/REPORT.md` + `.html` + `.pdf` + `.xlsx` + `.json` +
  `.csv`, charts in `reports/charts/*.svg`. Commit `reports/*` to publish.

## When — order matters

1. Transcripts **before** benchmarks (quality gate).
2. Warmup is automatic and discarded — never compare a first run.
3. Engine races need servers up **before** the bench command (connection
   refused = start the server; 404 = `--model` must equal the server's
   served name exactly).
4. `make_report.py` last — it degrades gracefully over whatever exists.

## Reading the report

- **Decode tok/s** is the headline for chat; **TTFT** for snappiness;
  **p95**, not the mean, tells you the worst normal experience.
- **Errors section:** failed runs are listed verbatim and excluded from
  means — a mean computed over partial data says so by its `n`.
- **Cross-hardware table:** blank on purpose. Same `prompts.txt` + same
  temperature on the other machine (NVIDIA/AMD/Intel/Snapdragon) is the
  only honest way to fill it — never vendor-published numbers.
- **Charts:** SVG in HTML/MD (`reports/charts/`), natively drawn bars in
  PDF — all three generated from the same numbers.

## Exit codes & interruptions

- `0` — finished; summary printed (builds, runs, files, elapsed, next step).
- `130` — Ctrl+C: partial `results/*.json` kept, re-run to continue.
- Anything else — real failure; the traceback names it.

## Reproducibility checklist (for published numbers)

- [ ] Machine, chip, RAM, OS recorded (automatic in every report header)
- [ ] Model IDs + quants recorded (automatic)
- [ ] Temperature 0 for races (automatic), 0.7 noted for transcripts
- [ ] Repeats ≥ 3 for anything you quote (default; `--repeats 1` is scouting)
- [ ] Background apps closed, plugged in, no Low Power Mode
- [ ] `reports/` committed alongside the claim
