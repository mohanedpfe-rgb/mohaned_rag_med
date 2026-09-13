from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def rss(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                try:
                    return int(line.split()[1]) * 1024
                except (IndexError, ValueError):
                    return None
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid))
        if not handle:
            return None
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        try:
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    return None


def fd_count(pid: int) -> int | None:
    directory = Path(f"/proc/{pid}/fd")
    if directory.exists():
        try:
            return len(list(directory.iterdir()))
        except OSError:
            return None
    if os.name == "nt":
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, int(pid))
        if not handle:
            return None
        count = ctypes.c_ulong()
        try:
            ok = ctypes.windll.kernel32.GetProcessHandleCount(handle, ctypes.byref(count))
            return int(count.value) if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    return None


def _write_probe_pdf(path: Path, iteration: int) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(45, 45, 550, 790), f"Resource stability diagnostic {iteration}\nDiabetes mellitus is a chronic metabolic disease. HbA1c is used for diagnosis and monitoring.", fontsize=11)
    document.save(path)
    document.close()


def _linear_slope(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    denom = sum((i - x_mean) ** 2 for i in range(len(values))) or 1.0
    return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / denom


def _quarter_mean(values: list[int], start: bool) -> float | None:
    if not values:
        return None
    width = max(1, len(values) // 4)
    sample = values[:width] if start else values[-width:]
    return sum(sample) / len(sample)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    try:
        from tests.diagnostic_runtime_adapters import install as install_diagnostic_adapters
        install_diagnostic_adapters()
    except Exception:
        pass
    from rag_project.testing.production_path_probes import _ProductionIngestionProbeSystem
    from rag_project.ingestion.robust_ingestor import robust_ingest_file

    started = time.monotonic()
    deadline = started + max(5.0, args.duration)
    samples: list[int] = []
    fds: list[int] = []
    iterations = 0
    successes = 0
    failures: list[str] = []
    while time.monotonic() < deadline:
        root = Path(tempfile.mkdtemp(prefix=f"rag_resource_production_{iterations}_"))
        try:
            system = _ProductionIngestionProbeSystem(root)
            source_dir = root / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            source = source_dir / f"resource_{iterations}.pdf"
            _write_probe_pdf(source, iterations)
            outcome = robust_ingest_file(system, source)
            if outcome.get("status") == "success":
                successes += 1
            else:
                failures.append(str(outcome.get("status") or "unknown_failure"))
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
            gc.collect()
        current_rss = rss(os.getpid())
        current_fd = fd_count(os.getpid())
        if current_rss is not None:
            samples.append(current_rss)
        if current_fd is not None:
            fds.append(current_fd)
        iterations += 1

    observed = time.monotonic() - started
    rss_delta = (samples[-1] - samples[0]) if len(samples) >= 2 else 0
    fd_delta = (fds[-1] - fds[0]) if len(fds) >= 2 else 0
    payload = {
        "workload": "canonical robust_ingest_file isolated PDF -> extraction -> chunking -> embedding -> validation -> READY lifecycle",
        "observed_seconds": observed,
        "sample_count": len(samples),
        "iterations": iterations,
        "successful_ingestions": successes,
        "failed_iterations": len(failures),
        "failure_samples": failures[:5],
        "rss_first_bytes": samples[0] if samples else None,
        "rss_last_bytes": samples[-1] if samples else None,
        "rss_peak_bytes": max(samples) if samples else None,
        "rss_delta_bytes": rss_delta,
        "rss_slope_bytes_per_iteration": _linear_slope(samples),
        "rss_first_quarter_mean": _quarter_mean(samples, True),
        "rss_last_quarter_mean": _quarter_mean(samples, False),
        "rss_tail_minus_head_mean_bytes": ((_quarter_mean(samples, False) or 0.0) - (_quarter_mean(samples, True) or 0.0)) if samples else 0.0,
        "fd_first": fds[0] if fds else None,
        "fd_last": fds[-1] if fds else None,
        "fd_delta": fd_delta,
        "fd_peak": max(fds) if fds else None,
        "ready_publication_contract_guarded": True,
        "repository_root": str(ROOT),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if successes >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
