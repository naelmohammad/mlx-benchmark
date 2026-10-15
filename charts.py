"""Pure-stdlib SVG bar charts for the lab reports.

WHAT THIS IS FOR: matplotlib is a heavy dependency for three bars. These
  charts render as inline SVG in REPORT.html, as .svg files referenced from
  REPORT.md, and the PDF renderer draws the same data with reportlab shapes.
  One data-prep helper (series_for) feeds all three outputs so the numbers
  cannot drift apart.
"""

from __future__ import annotations

import statistics


def pct(vals: list[float], p: float) -> float | None:
    """Percentile of a list (p in 0..100); None when empty."""
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 1)
    k = (len(vals) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


def dist(rows: list[dict], key: str) -> dict:
    """Mean/p50/p95/max/n for one metric across runs (nulls skipped)."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {"mean": round(statistics.mean(vals), 1) if vals else None,
            "p50": pct(vals, 50), "p95": pct(vals, 95),
            "max": round(max(vals), 1) if vals else None, "n": len(vals)}


def bar_chart(title: str, labels: list[str], values: list[float | None],
              unit: str, width: int = 640) -> str:
    """One horizontal bar chart as a self-contained SVG string."""
    rows = [(lb, v) for lb, v in zip(labels, values) if v is not None]
    if not rows:
        return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
                f"height='40'><text x='10' y='25' font-size='13' "
                f"fill='#888'>No data for {title}</text></svg>")
    top = max(v for _, v in rows)
    bar_h, gap, label_w = 26, 12, 250
    height = len(rows) * (bar_h + gap) + 44
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
             f"height='{height}' font-family='Helvetica,Arial,sans-serif'>",
             f"<text x='10' y='22' font-size='15' font-weight='bold' "
             f"fill='#222'>{title}</text>"]
    y = 36
    for label, value in rows:
        w = max(2, int((width - label_w - 90) * (value / top))) if top else 2
        short = label if len(label) <= 34 else label[:31] + "..."
        parts.append(
            f"<text x='10' y='{y + 18}' font-size='12' fill='#333'>{short}</text>"
            f"<rect x='{label_w}' y='{y}' width='{w}' height='{bar_h}' rx='4' "
            f"fill='#4F86C6'/><text x='{label_w + w + 8}' y='{y + 18}' "
            f"font-size='12' fill='#222'>{value} {unit}</text>")
        y += bar_h + gap
    parts.append("</svg>")
    return "".join(parts)
