from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import fitz


def rss(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists(): return None
    for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            try: return int(line.split()[1]) * 1024
            except (IndexError, ValueError): return None
    return None


def fd_count(pid: int) -> int | None:
    directory = Path(f"/proc/{pid}/fd")
    try: return len(list(directory.iterdir()))
    except OSError: return None


def _write_probe_pdf(path: Path, iteration: int) -> None:
    document = fitz.open(); page = document.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(45, 45, 550, 790), f"Resource stability diagnostic {iteration}\nDiabetes mellitus is a chronic metabolic disease. HbA1c is used for diagnosis and monitoring.", fontsize=11)
    document.save(path); document.close()


def _linear_slope(values: list[int]) -> float:
    if len(values) < 2: return 0.0
    x_mean = (len(values) - 1) / 2.0; y_mean = sum(values) / len(values); denom = sum((i - x_mean) ** 2 for i in range(len(values))) or 1.0
    return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / denom


def _quarter_mean(values: list[int], start: bool) -> float | None:
    if not values: return None
    width = max(1, len(values) // 4); sample = values[:width] if start else values[-width:]
    return sum(sample) / len(sample)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--duration", type=float, default=20.0); args = parser.parse_args()
    from rag_project.testing.production_path_probes import _ProductionIngestionProbeSystem
    from rag_project.ingestion.robust_ingestor import robust_ingest_file
    started = time.monotonic(); deadline = started + max(5.0, args.duration); samples: list[int] = []; fds: list[int] = []; iterations = 0; successes = 0
    while time.monotonic() < deadline:
        root = Path(tempfile.mkdtemp(prefix=f"rag_resource_production_{iterations}_"))
        try:
            system = _ProductionIngestionProbeSystem(root); source_dir = root / "source"; source_dir.mkdir(parents=True, exist_ok=True); source = source_dir / f"resource_{iterations}.pdf"; _write_probe_pdf(source, iterations); outcome = robust_ingest_file(system, source)
            if outcome.get("status") == "success": successes += 1
        finally:
            shutil.rmtree(root, ignore_errors=True); gc.collect()
        current_rss = rss(os.getpid()); current_fd = fd_count(os.getpid())
        if current_rss is not None: samples.append(current_rss)
        if current_fd is not None: fds.append(current_fd)
        iterations += 1
    observed = time.monotonic() - started
    payload = {
        "workload": "canonical robust_ingest_file isolated PDF -> extraction -> chunking -> embedding -> validation -> READY lifecycle",
        "observed_seconds": observed, "sample_count": len(samples), "iterations": iterations, "successful_ingestions": successes,
        "rss_first_bytes": samples[0] if samples else None, "rss_last_bytes": samples[-1] if samples else None, "rss_peak_bytes": max(samples) if samples else None,
        "rss_delta_bytes": (samples[-1] - samples[0]) if len(samples) >= 2 else None, "rss_slope_bytes_per_iteration": _linear_slope(samples),
        "rss_first_quarter_mean": _quarter_mean(samples, True), "rss_last_quarter_mean": _quarter_mean(samples, False),
        "rss_tail_minus_head_mean_bytes": (_quarter_mean(samples, False) - _quarter_mean(samples, True)) if samples else None,
        "fd_first": fds[0] if fds else None, "fd_last": fds[-1] if fds else None, "fd_delta": (fds[-1] - fds[0]) if len(fds) >= 2 else None,
        "fd_peak": max(fds) if fds else None,
    }
    print(json.dumps(payload, sort_keys=True)); return 0 if successes >= 3 else 1


if __name__ == "__main__": raise SystemExit(main())
