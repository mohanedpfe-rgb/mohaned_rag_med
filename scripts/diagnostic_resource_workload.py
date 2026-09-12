from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import fitz


def rss(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def fd_count(pid: int) -> int | None:
    directory = Path(f"/proc/{pid}/fd")
    try:
        return len(list(directory.iterdir()))
    except OSError:
        return None


def _write_probe_pdf(path: Path, iteration: int) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(45, 45, 550, 790),
        f"Resource stability diagnostic {iteration}\nDiabetes mellitus is a chronic metabolic disease. HbA1c is used for diagnosis and monitoring.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    from rag_project.testing.production_path_probes import _ProductionIngestionProbeSystem
    from rag_project.ingestion.robust_ingestor import robust_ingest_file

    started = time.monotonic()
    deadline = started + max(5.0, args.duration)
    samples: list[int] = []
    fds: list[int] = []
    iterations = 0
    successes = 0

    with tempfile.TemporaryDirectory(prefix="rag_resource_production_") as td:
        root = Path(td)
        system = _ProductionIngestionProbeSystem(root)
        source_dir = root / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        while time.monotonic() < deadline:
            source = source_dir / f"resource_{iterations}.pdf"
            _write_probe_pdf(source, iterations)
            outcome = robust_ingest_file(system, source)
            if outcome.get("status") == "success":
                successes += 1
            current_rss = rss(os.getpid())
            current_fd = fd_count(os.getpid())
            if current_rss is not None:
                samples.append(current_rss)
            if current_fd is not None:
                fds.append(current_fd)
            iterations += 1
            gc.collect()

    observed = time.monotonic() - started
    payload = {
        "workload": "canonical robust_ingest_file PDF -> extraction -> chunking -> embedding -> validation -> READY loop",
        "observed_seconds": observed,
        "sample_count": len(samples),
        "iterations": iterations,
        "successful_ingestions": successes,
        "rss_first_bytes": samples[0] if samples else None,
        "rss_last_bytes": samples[-1] if samples else None,
        "rss_peak_bytes": max(samples) if samples else None,
        "rss_delta_bytes": (samples[-1] - samples[0]) if len(samples) >= 2 else None,
        "fd_first": fds[0] if fds else None,
        "fd_last": fds[-1] if fds else None,
        "fd_delta": (fds[-1] - fds[0]) if len(fds) >= 2 else None,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if successes >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
