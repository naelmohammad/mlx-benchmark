"""Shared CLI helpers for the lab scripts.

WHAT THIS IS FOR: every script ends the same way — a completion summary
  (what ran, where the output went, how long it took, what to run next)
  instead of silence, and Ctrl+C keeps partial results with exit 130
  instead of a traceback.
"""

from __future__ import annotations

import sys
import time


def complete(label: str, details: list[str], started: float,
             next_step: str = "") -> None:
    mins, secs = divmod(time.perf_counter() - started, 60)
    print(f"\nDone: {label} complete in {int(mins)}m {secs:.1f}s.")
    for d in details:
        print(f"  {d}")
    if next_step:
        print(f"  Next: {next_step}")


def guarded(main_fn) -> None:
    """Run main_fn; on Ctrl+C keep partial results and exit 130."""
    try:
        main_fn()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results kept in results/; "
              "re-run to continue.", file=sys.stderr)
        sys.exit(130)
