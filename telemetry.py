"""System telemetry sampled alongside benchmark runs.

WHAT THIS IS FOR: puts CPU, memory (and GPU when obtainable) into the same
  run output, so a latency spike can be correlated with a pressure event
  without a separate profiling session.
WHAT IT CAN SEE: psutil gives CPU % and memory anytime, no privileges.
  GPU power/utilization on macOS needs `powermetrics`, which needs sudo:
  gpu_snapshot() tries passwordless sudo once per race and otherwise records
  exactly why GPU data is absent — never zeros pretending to be measurements.
"""

from __future__ import annotations

import plistlib
import subprocess


def sample() -> dict:
    """Cheap point-in-time CPU + memory sample (call per run)."""
    import psutil

    cpu = psutil.cpu_percent(interval=None)  # % since last call
    mem = psutil.virtual_memory()
    return {"cpu_percent": round(cpu, 1),
            "mem_used_gb": round(mem.used / 1e9, 2),
            "mem_percent": round(mem.percent, 1)}


def prime() -> None:
    """Seed psutil's CPU baseline; call once before a race."""
    import psutil

    psutil.cpu_percent(interval=None)


def gpu_snapshot() -> dict:
    """One GPU power/util snapshot, or the honest reason it's absent."""
    try:
        out = subprocess.run(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "500",
             "--samplers", "gpu_power", "-f", "plist"],
            capture_output=True, timeout=20)
    except FileNotFoundError:
        return {"unavailable": "powermetrics not found (macOS only)"}
    except subprocess.TimeoutExpired:
        return {"unavailable": "powermetrics timed out"}
    if out.returncode != 0:
        return {"unavailable": "needs sudo; run once with `sudo powermetrics` "
                               "to cache credentials, or skip GPU telemetry"}
    try:
        data = plistlib.loads(out.stdout)
        gpu = (data.get("gpu_power") or {})
        return {"power_w": gpu.get("combined_power")
                if isinstance(gpu, dict) else None,
                "raw": "powermetrics gpu_power (500ms sample)"}
    except Exception as e:  # noqa: BLE001 — reported, not raised
        return {"unavailable": f"plist parse failed: {e}"}
