"""Run a subprocess repeatedly while sampling memory and emit a measured artifact.

Certification requires a real 24-hour run. The default duration is therefore
24 hours; use --duration-seconds for a shorter local smoke measurement. No
short run is labeled as a 24-hour certification.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def _rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                try:
                    return int(line.split()[1]) * 1024
                except (IndexError, ValueError):
                    return None
    return None


def run(command: str, duration_seconds: float, sample_interval: float) -> dict[str, object]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("--command must not be empty")
    started = time.time()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples: list[int] = []
    try:
        deadline = time.monotonic() + max(0.1, duration_seconds)
        while process.poll() is None and time.monotonic() < deadline:
            rss = _rss_bytes(process.pid)
            if rss is not None:
                samples.append(rss)
            time.sleep(max(0.1, sample_interval))
        timed_out = process.poll() is None
        if timed_out:
            process.terminate()
        stdout, stderr = process.communicate(timeout=10)
    except Exception:
        process.kill()
        stdout, stderr = process.communicate()
        raise
    elapsed = time.time() - started
    first = samples[0] if samples else None
    last = samples[-1] if samples else None
    delta = (last - first) if first is not None and last is not None else None
    payload = {
        "created": time.time(),
        "command": command,
        "requested_duration_seconds": duration_seconds,
        "observed_duration_seconds": elapsed,
        "sample_interval_seconds": sample_interval,
        "sample_count": len(samples),
        "rss_first_bytes": first,
        "rss_last_bytes": last,
        "rss_delta_bytes": delta,
        "rss_peak_bytes": max(samples) if samples else None,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stderr_tail": stderr[-2000:],
        "stdout_tail": stdout[-2000:],
        "certification_24h": duration_seconds >= 86400 and elapsed >= 86400 and not timed_out and process.returncode == 0,
    }
    payload["passed"] = bool(payload["certification_24h"] and (delta is None or delta >= -50 * 1024 * 1024))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True, help="Command to run under observation")
    parser.add_argument("--duration-seconds", type=float, default=86400.0)
    parser.add_argument("--sample-interval", type=float, default=30.0)
    parser.add_argument("--output", default="artifacts/memory_24h.json")
    args = parser.parse_args()
    payload = run(args.command, args.duration_seconds, args.sample_interval)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
