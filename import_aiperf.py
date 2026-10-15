"""Import an aiperf JSON export into results/ for make_report.

WHAT THIS IS FOR: aiperf (NVIDIA's validator) profiles ANY OpenAI-compatible
  endpoint — mlx_lm.server, rapid-mlx, Ollama included. Run it externally,
  then import its aggregates here so our report tables show an independent
  second opinion next to our own numbers.
HONESTY RULE: aiperf exports aggregates, not per-run rows. This converter
  copies those aggregates verbatim into `aiperf_summary` and leaves `runs`
  empty — the report labels such rows as aiperf aggregates with their request
  count, and never mixes them into our per-run distributions.
INPUTS:  an aiperf export (profile_export_aiperf.json).
OUTPUTS: results/engines_aiperf-<tag>.json.

Usage:
    uvx --python 3.12 aiperf profile --model qwen3.5:4b-mlx \\
      --endpoint-type chat --streaming --url http://localhost:11434 \\
      --tokenizer Qwen/Qwen3.5-4B --num-requests 20 \\
      --output-artifact-dir /tmp/aiperf-run
    uv run python import_aiperf.py --export /tmp/aiperf-run/profile_export_aiperf.json \\
      --tag aiperf-qwen35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def g(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True,
                    help="aiperf profile_export_aiperf.json path.")
    ap.add_argument("--tag", required=True,
                    help="Tag, e.g. aiperf-qwen35 (file: engines_<tag>.json).")
    args = ap.parse_args()

    d = json.loads(Path(args.export).read_text())
    cfg = d.get("input_config", {})
    models = g(cfg, "models", "items", default=[]) or []
    urls = g(cfg, "endpoint", "urls", default=[]) or []
    phases = cfg.get("phases", []) or []
    prof = next((p for p in phases if p.get("name") == "profiling"), {})
    summary = {
        "ttft_s": round(g(d, "time_to_first_token", "avg", default=0) / 1000, 3),
        "itl_p50_ms": g(d, "inter_token_latency", "p50"),
        "decode_rate": g(d, "output_token_throughput", "avg"),
        "rate_unit": g(d, "output_token_throughput", "unit",
                       default="tokens/sec"),
        "rps": g(d, "request_throughput", "avg"),
        "request_count": g(d, "request_count", "avg"),
        "error_rate": g(d, "request_error_rate", "avg"),
        "concurrency": prof.get("concurrency", 1),
    }
    out = {"engine_tag": "aiperf",
           "model": models[0]["name"] if models else "unknown",
           "base_url": urls[0] if urls else "unknown",
           "repeats": summary["request_count"],
           "concurrency": summary["concurrency"],
           "aiperf_summary": summary,
           "aiperf_version": d.get("aiperf_version"),
           "cli_command": g(d, "run_info", "cli_command"),
           "export_path": str(args.export),
           "runs": []}
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"engines_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print(f"  model={out['model']} decode={summary['decode_rate']:.1f} "
          f"{summary['rate_unit']} itl_p50={summary['itl_p50_ms']:.2f}ms "
          f"n={summary['request_count']}")


if __name__ == "__main__":
    main()
