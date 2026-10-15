"""Merge every results/*.json into REPORT.md.

Usage:
    uv run python make_report.py [--report REPORT.md]

Reads bench_*.json (in-process mlx-lm numbers), inference_*.json (which
prompts were checked), and engines_*.json (server-vs-server race), and writes
a Markdown report with machine specs, on-disk sizes, and comparison tables.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 1) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="REPORT.md")
    args = ap.parse_args()

    results = ROOT / "results"
    info = sys_info()
    lines = ["# MLX benchmark report", ""]
    lines += [f"- Machine: {info['chip']} · {info['cpu_count']} CPUs · "
              f"{info['memory_gb']} GB ({info['os']})"]
    lines.append("")

    lines += ["## Models on disk", ""]
    for m in sorted((ROOT / "models").glob("*")) if (ROOT / "models").exists() else []:
        lines.append(f"- `{m.name}`: {dir_size_gb(m)} GB")
    if not (ROOT / "models").exists():
        lines.append("- (no models downloaded yet)")
    lines.append("")

    lines += ["## bf16 vs 4-bit (Apple mlx-lm, in-process)", ""]
    benches = sorted(results.glob("bench_*.json"))
    if benches:
        lines += ["| build | prefill tok/s | decode tok/s | peak mem GB |",
                  "|---|---|---|---|"]
        for b in benches:
            d = json.loads(b.read_text())
            runs = d["runs"]
            lines.append(f"| `{Path(d['model']).name}` | "
                         f"{mean(runs, 'prompt_tps')} | "
                         f"{mean(runs, 'generation_tps')} | "
                         f"{mean(runs, 'peak_memory_gb')} |")
    else:
        lines.append("No in-process benchmarks yet — run `uv run python bench.py`.")
    lines.append("")

    lines += ["## Engine race (OpenAI-compatible servers, same prompts)", ""]
    engines = sorted(results.glob("engines_*.json"))
    if engines:
        lines += ["| engine | model served | TTFT s | decode rate |",
                  "|---|---|---|---|"]
        for e in engines:
            d = json.loads(e.read_text())
            runs = d["runs"]
            rate = mean(runs, "tokens_per_sec")
            unit = "tok/s" if rate is not None else "chars/s"
            rate = rate if rate is not None else mean(runs, "chars_per_sec")
            lines.append(f"| {d['engine_tag']} | `{d['model']}` | "
                         f"{mean(runs, 'ttft_seconds')} s | {rate} {unit} |")
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
    infs = sorted(results.glob("inference_*.json"))
    if infs:
        for f in infs:
            d = json.loads(f.read_text())
            lines.append(f"- `{Path(d['model']).name}`: "
                         f"{len(d['cases'])} prompts answered "
                         f"(see `{f.relative_to(ROOT)}`).")
    else:
        lines.append("No inference transcripts yet — run "
                     "`uv run python inference_test.py`.")
    lines.append("")

    out = ROOT / args.report
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
