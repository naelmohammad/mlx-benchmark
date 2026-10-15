"""Step 7 of the lab: merge every results/*.json into reports.

WHAT THIS IS FOR: turns raw JSON into readable reports — machine specs
  (chip/RAM, because numbers without conditions are trivia), on-disk model
  sizes, the bf16-vs-4-bit table, and the engine-race table with its
  methodology footnote. Missing stages degrade gracefully ("not run yet").
INPUTS:  results/bench_*.json, results/inference_*.json, results/engines_*.json.
OUTPUTS: REPORT.md, REPORT.html, REPORT.pdf, REPORT.xlsx, REPORT.json,
  REPORT.csv (pick with --format; default writes all six).

Usage:
    uv run python make_report.py [--out REPORT] [--format all]
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import platform
import statistics
import subprocess
from pathlib import Path

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
    report: dict = {"machine": sys_info(), "models": [], "inprocess": [],
                    "race": [], "qualitative": [], "runs": []}

    if models_dir.exists():
        for m in sorted(models_dir.glob("*")):
            report["models"].append({"name": m.name, "size_gb": dir_size_gb(m)})

    for b in sorted(results.glob("bench_*.json")):
        d = json.loads(b.read_text())
        runs = d["runs"]
        build = Path(d["model"]).name
        report["inprocess"].append({
            "build": build, "repeats": d.get("repeats"),
            "prefill_tps": mean(runs, "prompt_tps"),
            "decode_tps": mean(runs, "generation_tps"),
            "peak_memory_gb": mean(runs, "peak_memory_gb"),
        })
        for r in runs:
            report["runs"].append({"kind": "inprocess", "file": b.name,
                                   "build": build, **r})

    for e in sorted(results.glob("engines_*.json")):
        d = json.loads(e.read_text())
        runs = d["runs"]
        rate = mean(runs, "tokens_per_sec")
        unit = "tok/s" if rate is not None else "chars/s"
        if rate is None:
            rate = mean(runs, "chars_per_sec")
        report["race"].append({
            "engine": d["engine_tag"], "model": d["model"],
            "repeats": d.get("repeats"), "ttft_s": mean(runs, "ttft_seconds"),
            "decode_rate": rate, "rate_unit": unit,
        })
        for r in runs:
            report["runs"].append({"kind": "engine", "file": e.name,
                                   "engine": d["engine_tag"],
                                   "model": d["model"], **r})

    for f in sorted(results.glob("inference_*.json")):
        d = json.loads(f.read_text())
        report["qualitative"].append({
            "model": Path(d["model"]).name,
            "cases": len(d["cases"]), "file": str(f.relative_to(ROOT)),
        })
    return report


def render_md(r: dict) -> str:
    info = r["machine"]
    lines = ["# MLX benchmark report", ""]
    lines += [f"- Machine: {info['chip']} · {info['cpu_count']} CPUs · "
              f"{info['memory_gb']} GB ({info['os']})"]
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
        lines += ["| build | prefill tok/s | decode tok/s | peak mem GB |",
                  "|---|---|---|---|"]
        for b in r["inprocess"]:
            lines.append(f"| `{b['build']}` | {b['prefill_tps']} | "
                         f"{b['decode_tps']} | {b['peak_memory_gb']} |")
    else:
        lines.append("No in-process benchmarks yet — run `uv run python bench.py`.")
    lines.append("")
    lines += ["## Engine race (OpenAI-compatible servers, same prompts)", ""]
    if r["race"]:
        lines += ["| engine | model served | TTFT s | decode rate |",
                  "|---|---|---|---|"]
        for e in r["race"]:
            lines.append(f"| {e['engine']} | `{e['model']}` | {e['ttft_s']} s | "
                         f"{e['decode_rate']} {e['rate_unit']} |")
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
    return "\n".join(lines)


def render_html(r: dict) -> str:
    info = r["machine"]
    esc = html.escape
    out = io.StringIO()
    out.write("<!DOCTYPE html><html><head><meta charset='utf-8'>"
              "<title>MLX benchmark report</title><style>"
              "body{font-family:-apple-system,Helvetica,Arial,sans-serif;"
              "max-width:900px;margin:2em auto;padding:0 1em;color:#222}"
              "table{border-collapse:collapse;width:100%;margin:1em 0}"
              "th,td{border:1px solid #ccc;padding:.4em .6em;text-align:left}"
              "th{background:#f4f4f4}code{background:#f4f4f4;padding:0 .3em}"
              ".note{color:#555;font-style:italic}</style></head><body>")
    out.write("<h1>MLX benchmark report</h1>")
    out.write(f"<p>Machine: {esc(str(info['chip']))} · "
              f"{esc(str(info['cpu_count']))} CPUs · "
              f"{esc(str(info['memory_gb']))} GB ({esc(str(info['os']))})</p>")
    out.write("<h2>Models on disk</h2>")
    if r["models"]:
        out.write("<ul>" + "".join(
            f"<li><code>{esc(m['name'])}</code>: {m['size_gb']} GB</li>"
            for m in r["models"]) + "</ul>")
    else:
        out.write("<p>(no models downloaded yet)</p>")
    out.write("<h2>bf16 vs 4-bit (Apple mlx-lm, in-process)</h2>")
    if r["inprocess"]:
        out.write("<table><tr><th>build</th><th>prefill tok/s</th>"
                  "<th>decode tok/s</th><th>peak mem GB</th></tr>")
        for b in r["inprocess"]:
            out.write(f"<tr><td><code>{esc(b['build'])}</code></td>"
                      f"<td>{b['prefill_tps']}</td><td>{b['decode_tps']}</td>"
                      f"<td>{b['peak_memory_gb']}</td></tr>")
        out.write("</table>")
    else:
        out.write("<p>No in-process benchmarks yet.</p>")
    out.write("<h2>Engine race</h2>")
    if r["race"]:
        out.write("<table><tr><th>engine</th><th>model served</th><th>TTFT s</th>"
                  "<th>decode rate</th></tr>")
        for e in r["race"]:
            out.write(f"<tr><td>{esc(e['engine'])}</td>"
                      f"<td><code>{esc(e['model'])}</code></td>"
                      f"<td>{e['ttft_s']}</td>"
                      f"<td>{e['decode_rate']} {e['rate_unit']}</td></tr>")
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
    out.write("</body></html>")
    return out.getvalue()


def render_json(r: dict) -> str:
    return json.dumps(r, indent=2)


def render_csv(r: dict) -> str:
    out = io.StringIO()
    cols = ["kind", "file", "build", "engine", "model", "prompt",
            "prompt_tokens", "prompt_tps", "generation_tokens",
            "generation_tps", "peak_memory_gb", "wall_seconds", "ttft_seconds",
            "output_chars", "chars_per_sec", "completion_tokens",
            "tokens_per_sec"]
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
        ws.append([b["build"], b["prefill_tps"], b["decode_tps"],
                   b["peak_memory_gb"]])
    ws.append([])
    ws.append(["Engine race"])
    ws.append(["engine", "model served", "TTFT s", "decode rate", "unit"])
    for c in ws[ws.max_row]:
        c.font = bold
    for e in r["race"]:
        ws.append([e["engine"], e["model"], e["ttft_s"], e["decode_rate"],
                   e["rate_unit"]])

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

    wb.save(path)


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
                       f"{info['memory_gb']} GB ({info['os']})",
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
    section("bf16 vs 4-bit (in-process)",
            ["build", "prefill tok/s", "decode tok/s", "peak mem GB"],
            [[b["build"], b["prefill_tps"], b["decode_tps"],
              b["peak_memory_gb"]] for b in r["inprocess"]],
            "No in-process benchmarks yet.")
    section("Engine race", ["engine", "model", "TTFT s", "decode rate"],
            [[e["engine"], e["model"], e["ttft_s"],
              f"{e['decode_rate']} {e['rate_unit']}"] for e in r["race"]],
            "No engine race yet.")
    section("Qualitative check", ["model", "prompts answered", "file"],
            [[q["model"], q["cases"], q["file"]] for q in r["qualitative"]],
            "No inference transcripts yet.")
    doc.build(story)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="REPORT",
                    help="Report base name (extensions added per format).")
    ap.add_argument("--format", default="all", choices=("all",) + FORMATS,
                    help="Which report to write (default: all six).")
    args = ap.parse_args()

    wanted = FORMATS if args.format == "all" else (args.format,)
    report = collect()
    base = ROOT / args.out
    for fmt in wanted:
        if fmt == "md":
            (base.with_suffix(".md")).write_text(render_md(report))
        elif fmt == "html":
            (base.with_suffix(".html")).write_text(render_html(report))
        elif fmt == "json":
            (base.with_suffix(".json")).write_text(render_json(report))
        elif fmt == "csv":
            (base.with_suffix(".csv")).write_text(render_csv(report))
        elif fmt == "xlsx":
            render_xlsx(report, base.with_suffix(".xlsx"))
        elif fmt == "pdf":
            render_pdf(report, base.with_suffix(".pdf"))
        print(f"wrote {base.with_suffix('.' + fmt)}")


if __name__ == "__main__":
    main()
