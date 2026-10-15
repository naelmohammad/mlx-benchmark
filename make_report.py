"""Step 7 of the lab: merge every results/*.json into reports.

WHAT THIS IS FOR: turns raw JSON into readable reports — machine specs
  (chip/RAM, because numbers without conditions are trivia), on-disk model
  sizes, the bf16-vs-4-bit table, and the engine-race table with its
  methodology footnote. Missing stages degrade gracefully ("not run yet").
INPUTS:  results/bench_*.json, results/inference_*.json, results/engines_*.json.
OUTPUTS: reports/REPORT.md, .html, .pdf, .xlsx, .json, .csv
  (pick with --format; default writes all six).

Usage:
    uv run python make_report.py [--out reports/REPORT] [--format all]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import html
import io
import json
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from charts import bar_chart, dist
from labutil import complete, guarded

ROOT = Path(__file__).resolve().parent
FORMATS = ("md", "html", "pdf", "xlsx", "json", "csv")


def sys_info() -> dict:
    def ctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "chip": ctl("machdep.cpu.brand_string"),
        "cpu_count": ctl("hw.ncpu"),
        "memory_gb": (
            round(int(ctl("hw.memsize")) / 1e9, 1)
            if ctl("hw.memsize").isdigit() else "unknown"
        ),
    }


def dir_size_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 2)


def mean(rows: list[dict], key: str):
    # Skips nulls (e.g. a metric one mlx-lm version doesn't report) so one
    # missing field can't sink a whole table; empty means "no data", not zero.
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 1) if vals else None


def collect() -> dict:
    """Read every results file into one plain-data report dict."""
    results = ROOT / "results"
    models_dir = ROOT / "models"
    prompts = [l.strip() for l in (ROOT / "prompts.txt").read_text().splitlines()
               if l.strip() and not l.startswith("#")] \
        if (ROOT / "prompts.txt").exists() else []
    disk = shutil.disk_usage(ROOT)
    report: dict = {
        "meta": {
            "generated_at": datetime.datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "prompts": len(prompts),
            "disk_free_gb": round(disk.free / 1e9, 1),
        },
        "machine": sys_info(), "models": [], "inprocess": [],
        "race": [], "qualitative": [], "errors": [], "runs": [],
    }

    if models_dir.exists():
        for m in sorted(models_dir.glob("*")):
            report["models"].append({"name": m.name, "size_gb": dir_size_gb(m)})

    for b in sorted(results.glob("bench_*.json")):
        if b.name.startswith("bench_shapes_"):
            continue  # handled by the single-request section below
        d = json.loads(b.read_text())
        runs = d["runs"]
        build = Path(d["model"]).name
        size = next((m["size_gb"] for m in report["models"]
                       if m["name"] == build), None)
        decode_mean = mean(runs, "generation_tps")
        entry = {
            "build": build, "repeats": d.get("repeats"),
            "prefill_tps": mean(runs, "prompt_tps"),
            "decode": dist(runs, "generation_tps"),
            "ttft": dist(runs, "wall_seconds"),
            "peak_memory_gb": mean(runs, "peak_memory_gb"),
            "prompt_tokens_mean": mean(runs, "prompt_tokens"),
            "gen_tokens_mean": mean(runs, "generation_tokens"),
            "throttle": throttle_factor(runs),
            "errors": sum(1 for r in runs if r.get("error")),
        }
        entry.update(flop_estimates(size, build, decode_mean))
        # Time-cost per 1M tokens at the measured rate (unit economics input;
        # multiply by $/GPU-hour on rented hardware for dollars).
        entry["sec_per_1m_tokens"] = (
            round(1e6 / decode_mean, 1) if decode_mean else None)
        report["inprocess"].append(entry)
        for r in runs:
            if r.get("error"):
                report["errors"].append({"file": b.name, "build": build,
                                         "prompt": r.get("prompt"),
                                         "error": r["error"]})
            report["runs"].append({"kind": "inprocess", "file": b.name,
                                   "build": build, **r})

    for e in sorted(results.glob("engines_*.json")):
        d = json.loads(e.read_text())
        if "runs" not in d:
            # Sweep frontier files carry points, not runs (see below).
            if "frontier" in d:
                report.setdefault("frontiers", []).append({
                    "engine": d["engine_tag"], "model": d["model"],
                    "levels": d.get("levels"), "points": d["frontier"]})
            continue
        runs = d["runs"]
        if not runs and "aiperf_summary" in d:
            # aiperf imports carry verbatim aggregates, not per-run rows:
            # display them labeled, never blend into our distributions.
            s = d["aiperf_summary"]
            report["race"].append({
                "engine": f"{d['engine_tag']}*",
                "model": d["model"], "repeats": d.get("repeats"),
                "concurrency": d.get("concurrency", 1),
                "ttft_s": s.get("ttft_s"), "ttft": {},
                "itl_p50_ms": (round(s["itl_p50_ms"], 2)
                               if s.get("itl_p50_ms") is not None else None),
                "decode_rate": (round(s["decode_rate"], 1)
                                if s.get("decode_rate") is not None else None),
                "rate_unit": s.get("rate_unit", "tokens/sec"),
                "decode": {}, "cpu_mean": None, "mem_gb_mean": None,
                "throttle": None, "gpu": None,
                "errors": None, "source": "aiperf aggregates",
                "aiperf_version": d.get("aiperf_version"),
            })
            continue
        rate = mean(runs, "tokens_per_sec")
        unit = "tok/s" if rate is not None else "chars/s"
        key = "tokens_per_sec" if rate is not None else "chars_per_sec"
        if rate is None:
            rate = mean(runs, "chars_per_sec")
        report["race"].append({
            "engine": d["engine_tag"], "model": d["model"],
            "repeats": d.get("repeats"),
            "concurrency": d.get("concurrency", 1),
            "ttft_s": mean(runs, "ttft_seconds"),
            "ttft": dist(runs, "ttft_seconds"),
            "itl_p50_ms": dist(runs, "itl_mean_ms")["p50"],
            "decode_rate": rate, "rate_unit": unit,
            "decode": dist(runs, key),
            "cpu_mean": mean(runs, "cpu_percent"),
            "mem_gb_mean": mean(runs, "mem_used_gb"),
            "throttle": throttle_factor(runs, "total_seconds"),
            "gpu": d.get("gpu"),
            "errors": sum(1 for r in runs if r.get("error")),
        })
        for r in runs:
            if r.get("error"):
                report["errors"].append({"file": e.name,
                                         "engine": d["engine_tag"],
                                         "model": d["model"],
                                         "prompt": r.get("prompt"),
                                         "error": r["error"]})
            report["runs"].append({"kind": "engine", "file": e.name,
                                   "engine": d["engine_tag"],
                                   "model": d["model"], **r})

    for f in sorted(results.glob("inference_*.json")):
        d = json.loads(f.read_text())
        report["qualitative"].append({
            "model": Path(d["model"]).name,
            "cases": len(d["cases"]), "file": str(f.relative_to(ROOT)),
        })

    for s in sorted(results.glob("bench_shapes_*.json")):
        d = json.loads(s.read_text())
        build = Path(d["model"]).name
        for row in d.get("shapes", []):
            report.setdefault("single", []).append({"build": build, **row})
    return report


def precision_bpp(name: str) -> float | None:
    """Bytes per parameter from a build dirname heuristic (labeled estimate)."""
    n = name.lower()
    if "4bit" in n or "4-bit" in n or "q4" in n:
        return 0.5
    if "8bit" in n or "8-bit" in n or "fp8" in n:
        return 1.0
    if "bf16" in n or "fp16" in n or "f16" in n:
        return 2.0
    return None


def flop_estimates(size_gb: float | None, build: str,
                   decode_tps: float | None) -> dict:
    """FLOPs / bandwidth / efficiency from weight bytes. Estimates, not
    measurements: params ≈ bytes ÷ bytes-per-param; forward pass ≈
    2 FLOP/param/token (dense; MoE does less — treat as an upper bound).
    Achieved bandwidth = bytes re-read per second at the measured decode
    rate; divide by your chip's published peak for utilization (MBU)."""
    if not size_gb:
        return {"precision_bpp": None, "est_params_b": None,
                "est_tflop_per_token": None, "achieved_bw_gbs": None,
                "tps_per_gb": None}
    bpp = precision_bpp(build)
    params = size_gb * 1e9 / bpp if bpp else None
    return {
        "precision_bpp": bpp,
        "est_params_b": round(params / 1e9, 1) if params else None,
        "est_tflop_per_token": round(2 * params / 1e12, 3) if params else None,
        "achieved_bw_gbs": (round(size_gb * decode_tps, 1)
                            if decode_tps else None),
        "tps_per_gb": (round(decode_tps / size_gb, 1)
                       if decode_tps and size_gb else None),
    }


def throttle_factor(runs: list[dict], key: str = "wall_seconds",
                    ) -> float | None:
    """Late-vs-early wall-time ratio: >1.05 sustained hints thermal throttling
    or background contention (proxy, not a sensor — see Power/Thermal)."""
    walls = [r[key] for r in runs
             if r.get(key) is not None and not r.get("error")]
    if len(walls) < 4:
        return None
    half = len(walls) // 2
    early = sum(walls[:half]) / half
    late = sum(walls[half:]) / (len(walls) - half)
    return round(late / early, 3) if early > 0 else None


def scatter_svg(title: str, points: list[tuple],
                xlabel: str, ylabel: str) -> str:
    """Pareto scatter: latency (x) vs throughput (y), concurrency-labeled."""
    W, H, pad = 640, 380, 56
    xs = [p[0] for p in points if p[0] is not None and p[1] is not None]
    ys = [p[1] for p in points if p[0] is not None and p[1] is not None]
    if not xs:
        return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' "
                f"height='60'><text x='10' y='30' font-size='13' fill='#888'>"
                f"No data for {title}</text></svg>")
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y1 = max(ys) * 1.1 or 1
    sx = lambda x: pad + (W - 2 * pad) * (x - x0) / (x1 - x0)
    sy = lambda y: H - pad - (H - 2 * pad) * (y / y1)
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' "
             f"height='{H}' font-family='Helvetica,Arial,sans-serif'>",
             f"<text x='10' y='22' font-size='15' font-weight='bold' "
             f"fill='#222'>{title}</text>",
             f"<line x1='{pad}' y1='{H - pad}' x2='{W - pad}' y2='{H - pad}' "
             f"stroke='#999'/><line x1='{pad}' y1='{pad}' x2='{pad}' "
             f"y2='{H - pad}' stroke='#999'/>",
             f"<text x='{W // 2}' y='{H - 8}' font-size='12' fill='#555' "
             f"text-anchor='middle'>{xlabel}</text>"]
    for x, y, label in points:
        if x is None or y is None:
            continue
        parts.append(
            f"<circle cx='{sx(x):.0f}' cy='{sy(y):.0f}' r='7' "
            f"fill='#4F86C6'/><text x='{sx(x):.0f}' y='{sy(y) - 12:.0f}' "
            f"font-size='11' fill='#222' text-anchor='middle'>c={label}</text>")
    parts.append("</svg>")
    return "".join(parts)


def triad_rows(r: dict) -> list[dict]:
    """Quality–Cost–Precision, one row per local build.

    Quality = transcript review status (human read, not a score — beware any
    table that scores quality it never measured). Cost = time per 1M tokens
    at the measured rate (× $/GPU-hour on rented hardware for dollars) plus
    GB footprint. Precision = weight bits (heuristic from build name)."""
    have_transcripts = {q["model"] for q in r["qualitative"]}
    rows = []
    for b in r["inprocess"]:
        rows.append({
            "build": b["build"],
            "quality": (f"transcripts read ({b['build']})"
                        if b["build"] in have_transcripts
                        else "NOT REVIEWED — run inference_test.py"),
            "sec_per_1m": b.get("sec_per_1m_tokens"),
            "size_gb": next((m["size_gb"] for m in r["models"]
                             if m["name"] == b["build"]), None),
            "bits": b.get("precision_bpp"),
            "decode": b["decode"]["mean"],
            "tps_per_gb": b.get("tps_per_gb"),
        })
    return rows


def chart_svgs(r: dict) -> dict:
    """Build the report charts; keys map to charts/<name>.svg files."""
    out = {
        "decode-tps": bar_chart(
            "Decode throughput (higher is better)",
            [b["build"] for b in r["inprocess"]],
            [b["decode"]["mean"] for b in r["inprocess"]], "tok/s"),
        "peak-mem": bar_chart(
            "Peak memory (lower is better)",
            [b["build"] for b in r["inprocess"]],
            [b["peak_memory_gb"] for b in r["inprocess"]], "GB"),
        "ttft": bar_chart(
            "Time to first token (lower is better)",
            [f"{e['engine']}:{e['model']}" for e in r["race"]],
            [e["ttft_s"] for e in r["race"]], "s"),
    }
    for f in r.get("frontiers", []):
        # Pareto: p50 latency (x) vs aggregate throughput (y), one dot per
        # concurrency level — the knee is the capacity answer.
        out[f"pareto-{f['engine']}"] = scatter_svg(
            f"Pareto frontier: {f['engine']}:{f['model']}",
            [(p["p50_latency_ms"], p["agg_tps"], p["concurrency"])
             for p in f["points"]],
            "p50 latency (ms) →", "aggregate TPS (tok/s) →")
    return out


CROSS_HW = """\
## Comparing across hardware (fill this in yourself)

A tok/s number only compares across machines when the workload is identical.
To add an NVIDIA / AMD / Intel / Snapdragon row to this report:

1. Copy `prompts.txt` to the other machine unchanged.
2. Serve any OpenAI-compatible endpoint there (vLLM or llama.cpp server on
   NVIDIA/AMD, llama.cpp on Intel iGPU, Ollama on Snapdragon X laptops).
3. Run `bench_engines.py --base-url <that server> --tag <gpu> --model <name>`
   here or there — the script only needs HTTP, not local weights.
4. Record: device, memory (VRAM/unified), quant, decode tok/s, TTFT s.

| device | memory | quant | decode tok/s | TTFT s |
|---|---|---|---|---|
| _this run: see tables above_ | | | | |

Never paste vendor-published numbers into this table: different prompts,
batch sizes, and contexts make them incomparable with yours.
"""


def render_md(r: dict, charts_dir: str = "charts") -> str:
    info = r["machine"]
    lines = ["# MLX benchmark report", ""]
    lines += [f"- Machine: {info['chip']} · {info['cpu_count']} CPUs · "
              f"{info['memory_gb']} GB ({info['os']})",
              f"- Generated: {r['meta']['generated_at']} · "
              f"{r['meta']['prompts']} prompts · "
              f"{r['meta']['disk_free_gb']} GB disk free"]
    lines.append("")
    lines += ["## Models on disk", ""]
    if r["models"]:
        for m in r["models"]:
            lines.append(f"- `{m['name']}`: {m['size_gb']} GB")
    else:
        lines.append("- (no models downloaded yet)")
    lines.append("")
    lines += ["## bf16 vs 4-bit (Apple mlx-lm, in-process)", ""]
    if r["inprocess"]:
        lines.append(f"![decode throughput]({charts_dir}/decode-tps.svg)")
        lines.append(f"![peak memory]({charts_dir}/peak-mem.svg)")
        lines.append("")
        lines += ["| build | prefill tok/s | decode tok/s (mean) | "
                  "tokens/run | peak mem GB |",
                  "|---|---|---|---|---|"]
        for b in r["inprocess"]:
            lines.append(f"| `{b['build']}` | {b['prefill_tps']} | "
                         f"{b['decode']['mean']} | {b['gen_tokens_mean']} | "
                         f"{b['peak_memory_gb']} |")
        lines += ["", "### Latency distribution (decode tok/s)", "",
                  "| build | mean | p50 | p95 | max | n |",
                  "|---|---|---|---|---|---|"]
        for b in r["inprocess"]:
            d = b["decode"]
            lines.append(f"| `{b['build']}` | {d['mean']} | {d['p50']} | "
                         f"{d['p95']} | {d['max']} | {d['n']} |")
    else:
        lines.append("No in-process benchmarks yet — run `uv run python bench.py`.")
    lines.append("")
    lines += ["## Engine race (OpenAI-compatible servers, same prompts)", ""]
    if r["race"]:
        lines += ["| engine | model served | TTFT s | ITL p50 ms | decode rate | "
                  "CPU % | mem GB |",
                  "|---|---|---|---|---|---|---|"]
        for e in r["race"]:
            lines.append(f"| {e['engine']} | `{e['model']}` | {e['ttft_s']} s | "
                         f"{e['itl_p50_ms']} | "
                         f"{e['decode_rate']} {e['rate_unit']} | "
                         f"{e['cpu_mean']} | {e['mem_gb_mean']} |")
        if any(e.get("source") == "aiperf aggregates" for e in r["race"]):
            lines.append("_Rows marked * are aiperf aggregates (independent "
                         "validator, different prompts), not our per-run "
                         "distributions._")
        lines += ["",
                  "_Methodology: same prompts, temperature 0, one warmup run "
                  "discarded per prompt, means over repeats. TTFT = "
                  "time-to-first-token over streamed SSE. Engines served "
                  "different (same-class) weights — see README step 6._"]
    else:
        lines.append("No engine race yet — run `uv run python bench_engines.py` "
                     "against each server (README step 6).")
    lines.append("")
    lines += ["## Qualitative check", ""]
    if r["qualitative"]:
        for q in r["qualitative"]:
            lines.append(f"- `{q['model']}`: {q['cases']} prompts answered "
                         f"(see `{q['file']}`).")
    else:
        lines.append("No inference transcripts yet — run "
                     "`uv run python inference_test.py`.")
    lines.append("")

    lines += ["## Errors & limits", ""]
    if r["errors"]:
        lines.append(f"{len(r['errors'])} run(s) errored "
                     "(excluded from means above):")
        for e in r["errors"][:20]:
            who = e.get("build") or f"{e.get('engine')}:{e.get('model')}"
            lines.append(f"- `{e['file']}` {who} “{e.get('prompt')}”: "
                         f"{e['error']}")
        if len(r["errors"]) > 20:
            lines.append(f"- …and {len(r['errors']) - 20} more (see JSON).")
    else:
        lines.append("Zero errored runs across every benchmark in this report.")
    lines += ["",
              "Limits of these numbers: batch size 1 (worst-case throughput); "
              "single machine (see header); greedy decoding (temperature 0) for "
              "engines, 0.7 for transcripts; KV-cache growth not isolated from "
              "weights in peak memory; prompt-caching effects excluded by "
              "fresh prompts per run.",
              ""]
    if r.get("single"):
        lines += ["## Single request results (fixed pp/tg workloads)", "",
                  "TTFT = prefill time; TPOT = ms per output token; PP TPS = "
                  "prefill rate; TG TPS = decode rate; Throughput = all "
                  "tokens ÷ wall (total utilization):", "",
                  "| test | build | TTFT ms | TPOT ms | PP TPS | TG TPS | "
                  "E2E s | Throughput | peak mem GB |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in r["single"]:
            lines.append(f"| {row['test']} | `{row['build']}` | "
                         f"{row['ttft_ms']} | {row['tpot_ms']} | "
                         f"{row['pp_tps']} | {row['tg_tps']} | {row['e2e_s']} | "
                         f"{row['throughput_tps']} | {row['peak_mem_gb']} |")
        lines.append("")
    if r.get("frontiers"):
        lines += ["## Continuous batching (Pareto frontier)", ""]
        for f in r["frontiers"]:
            lines.append(f"![pareto]({charts_dir}/pareto-{f['engine']}.svg)")
            lines.append("")
            base = next((p["agg_tps"] for p in f["points"]
                         if p.get("agg_tps")), None)
            lines += [f"### {f['engine']}:{f['model']}", "",
                      "| batch | TG TPS | speedup | PP TPS | PP TPS/req | "
                      "TTFT ms | E2E s | errors |",
                      "|---|---|---|---|---|---|---|---|"]
            for p in f["points"]:
                agg = p.get("agg_tps")
                lines.append(
                    f"| {p['concurrency']}x | {agg} | "
                    f"{(f'{agg / base:.2f}x' if agg and base else 'n/a')} | "
                    f"{p.get('pp_tps')} | {p.get('pp_tps_per_req')} | "
                    f"{p.get('ttft_mean_ms')} | {p.get('e2e_p50_s')} | "
                    f"{p['errors']} |")
            lines.append("")
        lines += ["Read it: TG TPS climbs with batch size, then flattens while "
                  "TTFT/E2E climb — the knee is the capacity answer. Speedup "
                  "is relative to the 1x row of the same sweep.", ""]
    lines += ["## Quality–Cost–Precision triad", "",
              "Quality is a human read of transcripts, never a computed score. "
              "Cost is time per 1M tokens at the measured rate (× $/GPU-hour "
              "for dollars). Precision is weight bits. Move one corner and "
              "the other two move:", "",
              "| build | quality | sec/1M tok | size GB | bits | "
              "decode tok/s | tok/s/GB |",
              "|---|---|---|---|---|---|---|"]
    for t in triad_rows(r):
        lines.append(f"| `{t['build']}` | {t['quality']} | {t['sec_per_1m']} | "
                     f"{t['size_gb']} | {t['bits']} | {t['decode']} | "
                     f"{t['tps_per_gb']} |")
    lines += ["",
              "FLOPs/analytics (estimates, dense-model upper bounds):"]
    for b in r["inprocess"]:
        lines.append(f"- `{b['build']}`: ~{b.get('est_params_b')}B params, "
                     f"~{b.get('est_tflop_per_token')} TFLOP/token, "
                     f"~{b.get('achieved_bw_gbs')} GB/s achieved bandwidth "
                     f"(÷ chip peak = utilization).")
    lines += ["",
              "## Power / Thermal vs Performance", "",
              "Throttle factor = late-run vs early-run wall time (>1.05 hints "
              "thermal throttling or background contention — a proxy, not a "
              "sensor):"]
    for b in r["inprocess"]:
        lines.append(f"- `{b['build']}`: throttle ×{b.get('throttle')}")
    for e in r["race"]:
        if e.get("source"):
            continue  # aiperf imports carry no throttle/telemetry series
        gpu = e.get("gpu") or {}
        g = (f"{gpu.get('power_w')} W" if gpu.get("power_w") is not None
             else gpu.get("unavailable", "no GPU telemetry"))
        lines.append(f"- {e['engine']}:{e['model']}: throttle "
                     f"×{e.get('throttle')}, GPU: {g}")
    lines += ["",
              "To measure power properly: one `sudo powermetrics` run during "
              "a race (GPU power + thermal pressure need root on macOS). "
              "Without it, watts-per-token and tokens/joule stay blank — "
              "this report leaves them blank rather than inventing them.",
              ""]
    lines.append(CROSS_HW)
    return "\n".join(lines)


def render_html(r: dict, svgs: dict | None = None) -> str:
    info = r["machine"]
    esc = html.escape
    svgs = svgs or {}
    out = io.StringIO()
    out.write("<!DOCTYPE html><html><head><meta charset='utf-8'>"
              "<title>MLX benchmark report</title><style>"
              "body{font-family:-apple-system,Helvetica,Arial,sans-serif;"
              "max-width:900px;margin:2em auto;padding:0 1em;color:#222}"
              "table{border-collapse:collapse;width:100%;margin:1em 0}"
              "th,td{border:1px solid #ccc;padding:.4em .6em;text-align:left}"
              "th{background:#f4f4f4}code{background:#f4f4f4;padding:0 .3em}"
              ".note{color:#555;font-style:italic}"
              "svg{max-width:100%;height:auto}</style></head><body>")
    out.write("<h1>MLX benchmark report</h1>")
    out.write(f"<p>Machine: {esc(str(info['chip']))} · "
              f"{esc(str(info['cpu_count']))} CPUs · "
              f"{esc(str(info['memory_gb']))} GB ({esc(str(info['os']))})<br>"
              f"Generated: {esc(r['meta']['generated_at'])} · "
              f"{r['meta']['prompts']} prompts · "
              f"{r['meta']['disk_free_gb']} GB disk free</p>")
    out.write("<h2>Models on disk</h2>")
    if r["models"]:
        out.write("<ul>" + "".join(
            f"<li><code>{esc(m['name'])}</code>: {m['size_gb']} GB</li>"
            for m in r["models"]) + "</ul>")
    else:
        out.write("<p>(no models downloaded yet)</p>")
    out.write("<h2>bf16 vs 4-bit (Apple mlx-lm, in-process)</h2>")
    if r["inprocess"]:
        if svgs.get("decode-tps"):
            out.write(svgs["decode-tps"])
        if svgs.get("peak-mem"):
            out.write(svgs["peak-mem"])
        out.write("<table><tr><th>build</th><th>prefill tok/s</th>"
                  "<th>decode tok/s</th><th>tokens/run</th>"
                  "<th>peak mem GB</th></tr>")
        for b in r["inprocess"]:
            out.write(f"<tr><td><code>{esc(b['build'])}</code></td>"
                      f"<td>{b['prefill_tps']}</td>"
                      f"<td>{b['decode']['mean']}</td>"
                      f"<td>{b['gen_tokens_mean']}</td>"
                      f"<td>{b['peak_memory_gb']}</td></tr>")
        out.write("</table>")
        out.write("<h3>Latency distribution (decode tok/s)</h3>"
                  "<table><tr><th>build</th><th>mean</th><th>p50</th>"
                  "<th>p95</th><th>max</th><th>n</th></tr>")
        for b in r["inprocess"]:
            d = b["decode"]
            out.write(f"<tr><td><code>{esc(b['build'])}</code></td>"
                      f"<td>{d['mean']}</td><td>{d['p50']}</td>"
                      f"<td>{d['p95']}</td><td>{d['max']}</td>"
                      f"<td>{d['n']}</td></tr>")
        out.write("</table>")
    else:
        out.write("<p>No in-process benchmarks yet.</p>")
    out.write("<h2>Engine race</h2>")
    if r["race"]:
        if svgs.get("ttft"):
            out.write(svgs["ttft"])
        out.write("<table><tr><th>engine</th><th>model served</th><th>TTFT s</th>"
                  "<th>ITL p50 ms</th><th>decode rate</th>"
                  "<th>CPU %</th><th>mem GB</th></tr>")
        for e in r["race"]:
            out.write(f"<tr><td>{esc(e['engine'])}</td>"
                      f"<td><code>{esc(e['model'])}</code></td>"
                      f"<td>{e['ttft_s']}</td><td>{e['itl_p50_ms']}</td>"
                      f"<td>{e['decode_rate']} {e['rate_unit']}</td>"
                      f"<td>{e['cpu_mean']}</td><td>{e['mem_gb_mean']}</td></tr>")
        out.write("</table><p class='note'>Methodology: same prompts, "
                  "temperature 0, one warmup run discarded per prompt, means "
                  "over repeats.</p>")
    else:
        out.write("<p>No engine race yet.</p>")
    out.write("<h2>Qualitative check</h2>")
    if r["qualitative"]:
        out.write("<ul>" + "".join(
            f"<li><code>{esc(q['model'])}</code>: {q['cases']} prompts "
            f"answered (see <code>{esc(q['file'])}</code>)</li>"
            for q in r["qualitative"]) + "</ul>")
    else:
        out.write("<p>No inference transcripts yet.</p>")
    out.write("<h2>Errors &amp; limits</h2>")
    if r["errors"]:
        out.write(f"<p>{len(r['errors'])} run(s) errored (excluded from "
                  "means):</p><ul>" + "".join(
            f"<li><code>{esc(e.get('file', ''))}</code> "
            f"{esc(e.get('build') or e.get('model', ''))}: "
            f"{esc(e['error'])}</li>" for e in r["errors"][:20]) + "</ul>")
    else:
        out.write("<p>Zero errored runs in this report.</p>")
    out.write("<p class='note'>Limits: batch size 1; single machine; greedy "
              "decoding for engines; KV-cache growth not isolated from weights "
              "in peak memory.</p>")
    if r.get("single"):
        out.write("<h2>Single request results (fixed pp/tg workloads)</h2>"
                  "<table><tr><th>test</th><th>build</th><th>TTFT ms</th>"
                  "<th>TPOT ms</th><th>PP TPS</th><th>TG TPS</th><th>E2E s</th>"
                  "<th>Throughput</th><th>peak mem GB</th></tr>")
        for row in r["single"]:
            out.write(f"<tr><td>{esc(row['test'])}</td>"
                      f"<td><code>{esc(row['build'])}</code></td>"
                      f"<td>{row['ttft_ms']}</td><td>{row['tpot_ms']}</td>"
                      f"<td>{row['pp_tps']}</td><td>{row['tg_tps']}</td>"
                      f"<td>{row['e2e_s']}</td>"
                      f"<td>{row['throughput_tps']}</td>"
                      f"<td>{row['peak_mem_gb']}</td></tr>")
        out.write("</table>")
    if r.get("frontiers"):
        out.write("<h2>Continuous batching (Pareto frontier)</h2>")
        for f in r["frontiers"]:
            key = f"pareto-{f['engine']}"
            if svgs.get(key):
                out.write(svgs[key])
            base = next((p["agg_tps"] for p in f["points"]
                         if p.get("agg_tps")), None)
            out.write(f"<h3>{esc(f['engine'])}:{esc(f['model'])}</h3>"
                      "<table><tr><th>batch</th><th>TG TPS</th><th>speedup</th>"
                      "<th>PP TPS</th><th>PP TPS/req</th><th>TTFT ms</th>"
                      "<th>E2E s</th><th>errors</th></tr>")
            for p in f["points"]:
                agg = p.get("agg_tps")
                out.write(f"<tr><td>{p['concurrency']}x</td><td>{agg}</td>"
                          f"<td>{(f'{agg / base:.2f}x' if agg and base else 'n/a')}</td>"
                          f"<td>{p.get('pp_tps')}</td>"
                          f"<td>{p.get('pp_tps_per_req')}</td>"
                          f"<td>{p.get('ttft_mean_ms')}</td>"
                          f"<td>{p.get('e2e_p50_s')}</td>"
                          f"<td>{p['errors']}</td></tr>")
            out.write("</table>")
    out.write("<h2>Quality–Cost–Precision triad</h2><p class='note'>Quality is "
              "a human read of transcripts, never a computed score. Cost is "
              "time per 1M tokens (× $/GPU-hour for dollars).</p>"
              "<table><tr><th>build</th><th>quality</th><th>sec/1M tok</th>"
              "<th>size GB</th><th>bits</th><th>decode tok/s</th>"
              "<th>tok/s/GB</th></tr>")
    for t in triad_rows(r):
        out.write(f"<tr><td><code>{esc(t['build'])}</code></td>"
                  f"<td>{esc(t['quality'])}</td><td>{t['sec_per_1m']}</td>"
                  f"<td>{t['size_gb']}</td><td>{t['bits']}</td>"
                  f"<td>{t['decode']}</td><td>{t['tps_per_gb']}</td></tr>")
    out.write("</table><h2>Power / Thermal vs Performance</h2>"
              "<p class='note'>Throttle = late vs early wall time; >1.05 hints "
              "throttling or contention (proxy, not sensor).</p><ul>")
    for b in r["inprocess"]:
        out.write(f"<li><code>{esc(b['build'])}</code>: throttle "
                  f"×{b.get('throttle')}</li>")
    for e in r["race"]:
        gpu = e.get("gpu") or {}
        g = (f"{gpu.get('power_w')} W" if gpu.get("power_w") is not None
             else gpu.get("unavailable", "no GPU telemetry"))
        out.write(f"<li>{esc(e['engine'])}:{esc(e['model'])}: throttle "
                  f"×{e.get('throttle')}, GPU: {esc(g)}</li>")
    out.write("</ul>")
    out.write("<h2>Comparing across hardware</h2><p>Copy <code>prompts.txt</code> "
              "to the other machine, serve any OpenAI-compatible endpoint "
              "(vLLM/llama.cpp on NVIDIA/AMD, llama.cpp on Intel, Ollama on "
              "Snapdragon X), and run <code>bench_engines.py --base-url "
              "&lt;server&gt;</code> with the same prompts. Record device, "
              "memory, quant, decode tok/s, TTFT. Never paste vendor-published "
              "numbers next to yours — different workloads make them "
              "incomparable.</p>")
    out.write("</body></html>")
    return out.getvalue()


def render_json(r: dict) -> str:
    return json.dumps(r, indent=2)


def render_csv(r: dict) -> str:
    out = io.StringIO()
    cols = ["kind", "file", "build", "engine", "model", "prompt",
            "prompt_tokens", "prompt_tps", "generation_tokens",
            "generation_tps", "peak_memory_gb", "wall_seconds", "ttft_seconds",
            "ttft_to", "output_chars", "chars_per_sec", "completion_tokens",
            "tokens_per_sec", "error"]
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for run in r["runs"]:
        w.writerow(run)
    return out.getvalue()


def render_xlsx(r: dict, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    bold = Font(bold=True)

    ws = wb.active
    ws.title = "Summary"
    info = r["machine"]
    ws.append(["MLX benchmark report"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([f"Machine: {info['chip']} · {info['cpu_count']} CPUs · "
               f"{info['memory_gb']} GB ({info['os']})"])
    ws.append([])
    ws.append(["Model on disk", "Size GB"])
    for c in ws[ws.max_row]:
        c.font = bold
    for m in r["models"]:
        ws.append([m["name"], m["size_gb"]])
    ws.append([])
    ws.append(["In-process summary"])
    ws.append(["build", "prefill tok/s", "decode tok/s", "peak mem GB"])
    for c in ws[ws.max_row]:
        c.font = bold
    for b in r["inprocess"]:
        ws.append([b["build"], b["prefill_tps"], b["decode"]["mean"],
                   b["peak_memory_gb"]])
    ws.append([])
    ws.append(["Engine race"])
    ws.append(["engine", "model served", "TTFT s", "ITL p50 ms", "decode rate",
               "unit", "CPU %", "mem GB"])
    for c in ws[ws.max_row]:
        c.font = bold
    for e in r["race"]:
        ws.append([e["engine"], e["model"], e["ttft_s"], e["itl_p50_ms"],
                   e["decode_rate"], e["rate_unit"], e["cpu_mean"],
                   e["mem_gb_mean"]])
    ws.append([])
    ws.append(["Latency distribution (decode tok/s: mean/p50/p95/max/n)"])
    ws.append(["target", "mean", "p50", "p95", "max", "n"])
    for c in ws[ws.max_row]:
        c.font = bold
    for b in r["inprocess"]:
        d = b["decode"]
        ws.append([b["build"], d["mean"], d["p50"], d["p95"], d["max"],
                   d["n"]])
    for e in r["race"]:
        d = e["decode"] or {}
        if d.get("mean") is None and e.get("source"):
            # aiperf imports have aggregates, not distributions: one labeled row.
            ws.append([f"{e['engine']}:{e['model']}",
                       f"aiperf avg {e.get('decode_rate')} "
                       f"{e.get('rate_unit')} (n={e.get('repeats')})"])
            continue
        ws.append([f"{e['engine']}:{e['model']}", d.get("mean"), d.get("p50"),
                   d.get("p95"), d.get("max"), d.get("n")])
    if r["errors"]:
        ws.append([])
        ws.append([f"Errors: {len(r['errors'])} errored run(s), see JSON/CSV"])
    ws.append([f"Generated {r['meta']['generated_at']} · "
               f"{r['meta']['prompts']} prompts · "
               f"{r['meta']['disk_free_gb']} GB disk free"])

    for sheet, rows in (("In-process runs",
                         [x for x in r["runs"] if x["kind"] == "inprocess"]),
                        ("Engine runs",
                         [x for x in r["runs"] if x["kind"] == "engine"])):
        w2 = wb.create_sheet(sheet)
        if not rows:
            w2.append(["(no data — run the corresponding benchmark first)"])
            continue
        cols = sorted({k for row in rows for k in row})
        w2.append(cols)
        for c in w2[1]:
            c.font = bold
        for row in rows:
            w2.append([json.dumps(v) if isinstance(v, (dict, list)) else v
                       for v in (row.get(c) for c in cols)])
        w2.freeze_panes = "A2"
        w2.auto_filter.ref = w2.dimensions

    if r.get("single"):
        ws1 = wb.create_sheet("Single request")
        ws1.append(["test", "build", "TTFT ms", "TPOT ms", "PP TPS", "TG TPS",
                    "E2E s", "Throughput", "peak GB"])
        for c in ws1[1]:
            c.font = bold
        for s in r["single"]:
            ws1.append([s["test"], s["build"], s["ttft_ms"], s["tpot_ms"],
                        s["pp_tps"], s["tg_tps"], s["e2e_s"],
                        s["throughput_tps"], s["peak_mem_gb"]])
        ws1.freeze_panes = "A2"
        ws1.auto_filter.ref = ws1.dimensions
    for f in r.get("frontiers", []):
        base = next((p["agg_tps"] for p in f["points"]
                     if p.get("agg_tps")), None)
        w3 = wb.create_sheet(f"Batching {f['engine']}"[:31])
        w3.append(["batch", "TG TPS", "speedup", "PP TPS", "PP TPS/req",
                   "TTFT ms", "E2E s", "errors"])
        for c in w3[1]:
            c.font = bold
        for p in f["points"]:
            w3.append([f"{p['concurrency']}x", p["agg_tps"],
                       (f"{p['agg_tps'] / base:.2f}x"
                        if p.get("agg_tps") and base else "n/a"),
                       p.get("pp_tps"), p.get("pp_tps_per_req"),
                       p.get("ttft_mean_ms"), p.get("e2e_p50_s"),
                       p["errors"]])
    wt = wb.create_sheet("Triad")
    wt.append(["build", "quality", "sec/1M tok", "size GB", "bits",
               "decode tok/s", "tok/s/GB"])
    for c in wt[1]:
        c.font = bold
    for t in triad_rows(r):
        wt.append([t["build"], t["quality"], t["sec_per_1m"], t["size_gb"],
                   t["bits"], t["decode"], t["tps_per_gb"]])

    wb.save(path)


def pdf_bars(story, styles, title: str, labels: list[str],
             values: list, unit: str) -> None:
    """One horizontal bar chart drawn with reportlab shapes (no matplotlib)."""
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer

    pairs = [(lb, v) for lb, v in zip(labels, values) if v is not None]
    story.append(Paragraph(title, styles["Heading3"]))
    if not pairs:
        story.append(Paragraph("No data.", styles["Normal"]))
        return
    top = max(v for _, v in pairs)
    W = 7.0 * inch
    d = Drawing(W, 22 * len(pairs) + 8)
    y = 0
    for label, value in pairs:
        w = max(4, (W - 2.6 * inch) * (value / top)) if top else 4
        d.add(String(0, y + 5, (label[:32] + "…") if len(label) > 33 else label,
                     fontSize=7))
        d.add(Rect(2.4 * inch, y, w, 13, fillColor="#4F86C6",
                   strokeColor="#4F86C6"))
        d.add(String(2.4 * inch + w + 4, y + 5, f"{value} {unit}", fontSize=7))
        y += 22
    story.append(d)
    story.append(Spacer(1, .1 * inch))


def render_pdf(r: dict, path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    info = r["machine"]
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            leftMargin=.7 * inch, rightMargin=.7 * inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("MLX benchmark report", styles["Title"]),
             Paragraph(f"Machine: {info['chip']} · {info['cpu_count']} CPUs · "
                       f"{info['memory_gb']} GB ({info['os']})<br/>"
                       f"Generated: {r['meta']['generated_at']} · "
                       f"{r['meta']['prompts']} prompts · "
                       f"{r['meta']['disk_free_gb']} GB disk free",
                       styles["Normal"]),
             Spacer(1, .2 * inch)]
    grid = TableStyle([("GRID", (0, 0), (-1, -1), .5, "#999"),
                       ("BACKGROUND", (0, 0), (-1, 0), "#eee"),
                       ("FONTSIZE", (0, 0), (-1, -1), 8)])

    def section(title: str, header: list, rows: list, empty: str):
        story.append(Paragraph(title, styles["Heading2"]))
        if rows:
            t = Table([header] + [[str(c) for c in row] for row in rows],
                      repeatRows=1)
            t.setStyle(grid)
            story.append(t)
        else:
            story.append(Paragraph(empty, styles["Normal"]))
        story.append(Spacer(1, .15 * inch))

    section("Models on disk", ["model", "size GB"],
            [[m["name"], m["size_gb"]] for m in r["models"]],
            "No models downloaded yet.")
    pdf_bars(story, styles, "Decode throughput, tok/s (higher is better)",
             [b["build"] for b in r["inprocess"]],
             [b["decode"]["mean"] for b in r["inprocess"]], "tok/s")
    section("bf16 vs 4-bit (in-process)",
            ["build", "prefill tok/s", "decode mean", "p50", "p95",
             "tokens/run", "peak mem GB"],
            [[b["build"], b["prefill_tps"], b["decode"]["mean"],
              b["decode"]["p50"], b["decode"]["p95"], b["gen_tokens_mean"],
              b["peak_memory_gb"]] for b in r["inprocess"]],
            "No in-process benchmarks yet.")
    pdf_bars(story, styles, "Time to first token, s (lower is better)",
             [f"{e['engine']}:{e['model']}" for e in r["race"]],
             [e["ttft_s"] for e in r["race"]], "s")
    section("Engine race",
            ["engine", "model", "TTFT s", "ITL p50 ms", "decode rate",
             "CPU %", "mem GB"],
            [[e["engine"], e["model"], e["ttft_s"], e["itl_p50_ms"],
              f"{e['decode_rate']} {e['rate_unit']}", e["cpu_mean"],
              e["mem_gb_mean"]] for e in r["race"]],
            "No engine race yet.")
    section("Qualitative check", ["model", "prompts answered", "file"],
            [[q["model"], q["cases"], q["file"]] for q in r["qualitative"]],
            "No inference transcripts yet.")
    if r["errors"]:
        section("Errors (excluded from means)",
                ["file", "target", "error"],
                [[e.get("file", ""),
                  e.get("build") or e.get("model", ""),
                  str(e["error"])[:100]] for e in r["errors"][:20]],
                "")
    else:
        story.append(Paragraph("Errors: zero errored runs in this report.",
                               styles["Normal"]))
        story.append(Spacer(1, .15 * inch))
    if r.get("single"):
        section("Single request results",
                ["test", "build", "TTFT ms", "TPOT ms", "PP TPS", "TG TPS",
                 "E2E s", "Throughput", "peak GB"],
                [[s["test"], s["build"], s["ttft_ms"], s["tpot_ms"],
                  s["pp_tps"], s["tg_tps"], s["e2e_s"],
                  s["throughput_tps"], s["peak_mem_gb"]]
                 for s in r["single"]],
                "No shapes run (bench.py --shapes).")
    for f in r.get("frontiers", []):
        base = next((p["agg_tps"] for p in f["points"]
                     if p.get("agg_tps")), None)
        section(f"Continuous batching: {f['engine']}:{f['model']}",
                ["batch", "TG TPS", "speedup", "PP TPS", "PP TPS/req",
                 "TTFT ms", "E2E s", "errors"],
                [[f"{p['concurrency']}x", p["agg_tps"],
                  (f"{p['agg_tps'] / base:.2f}x"
                   if p.get("agg_tps") and base else "n/a"),
                  p.get("pp_tps"), p.get("pp_tps_per_req"),
                  p.get("ttft_mean_ms"), p.get("e2e_p50_s"), p["errors"]]
                 for p in f["points"]],
                "No sweep run.")
    section("Quality–Cost–Precision triad",
            ["build", "quality", "sec/1M", "GB", "bits", "tok/s",
             "tok/s/GB"],
            [[t["build"], t["quality"], t["sec_per_1m"], t["size_gb"],
              t["bits"], t["decode"], t["tps_per_gb"]]
             for t in triad_rows(r)],
            "No local benchmarks yet.")
    story.append(Paragraph("Power / Thermal vs Performance", styles["Heading2"]))
    story.append(Paragraph("Throttle = late vs early wall time; >1.05 hints "
                           "throttling or contention (proxy, not sensor). "
                           "True watts need sudo powermetrics.", styles["Normal"]))
    for b in r["inprocess"]:
        story.append(Paragraph(f"{b['build']}: throttle ×{b.get('throttle')}",
                               styles["Normal"]))
    story.append(Spacer(1, .15 * inch))
    story.append(Paragraph(
        "Limits: batch size 1; single machine; greedy decoding for engines; "
        "KV-cache growth not isolated from weights in peak memory. "
        "Cross-hardware rule: same prompts.txt, same temperature, record "
        "device + memory + quant — never paste vendor-published numbers "
        "next to yours.", styles["Normal"]))
    doc.build(story)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/REPORT",
                    help="Report base path (extensions added per format).")
    ap.add_argument("--format", default="all", choices=("all",) + FORMATS,
                    help="Which report to write (default: all six).")
    args = ap.parse_args()
    started = time.perf_counter()

    wanted = FORMATS if args.format == "all" else (args.format,)
    report = collect()
    base = ROOT / args.out
    base.parent.mkdir(parents=True, exist_ok=True)
    # Charts live beside the reports; md references them relatively, html
    # inlines the same SVG strings, pdf redraws the same data natively.
    charts_dir = base.parent / "charts"
    svgs = chart_svgs(report)
    if {"md", "html"} & set(wanted):
        charts_dir.mkdir(parents=True, exist_ok=True)
        for name, svg in svgs.items():
            (charts_dir / f"{name}.svg").write_text(svg)
    for fmt in wanted:
        if fmt == "md":
            (base.with_suffix(".md")).write_text(render_md(report))
        elif fmt == "html":
            (base.with_suffix(".html")).write_text(render_html(report, svgs))
        elif fmt == "json":
            (base.with_suffix(".json")).write_text(render_json(report))
        elif fmt == "csv":
            (base.with_suffix(".csv")).write_text(render_csv(report))
        elif fmt == "xlsx":
            render_xlsx(report, base.with_suffix(".xlsx"))
        elif fmt == "pdf":
            render_pdf(report, base.with_suffix(".pdf"))
        print(f"wrote {base.with_suffix('.' + fmt)}")
    complete("Report",
             [f"{len(wanted)} format(s): "
              f"{', '.join(str(base.with_suffix('.' + f)) for f in wanted)}"],
             started, next_step="open reports/REPORT.html, or commit reports/*")


if __name__ == "__main__":
    guarded(main)
