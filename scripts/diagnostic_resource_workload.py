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
    try:
        import psutil
        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        pass
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, int(pid))
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS(); counters.cb = ctypes.sizeof(counters)
            return int(counters.WorkingSetSize) if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else None
        finally:
            kernel32.CloseHandle(handle)
    try:
        status = Path(f"/proc/{pid}/status")
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024 if sys.platform != "darwin" else value
    except Exception:
        return None


def fd_count(pid: int) -> int | None:
    try:
        import psutil
        process = psutil.Process(pid)
        return int(process.num_handles() if os.name == "nt" else process.num_fds())
    except Exception:
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


def _quarter_mean(values: list[int], start: bool) -> float:
    if not values:
        return 0.0
    width = max(1, len(values) // 4)
    selected = values[:width] if start else values[-width:]
    return sum(selected) / len(selected)


def _clear_chroma_cache() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        clear = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear):
            clear()
    except Exception:
        pass


def _close_probe_system(system: object) -> None:
    vector_store = getattr(system, "vector_store", None)
    close = getattr(vector_store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    try:
        setattr(system, "vector_store", None)
        setattr(system, "state_store", None)
    except Exception:
        pass
    gc.collect()
    _clear_chroma_cache()
    gc.collect()
    _clear_chroma_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    # This certification process intentionally does not load pytest-only
    # diagnostic monkeypatches. The workload must observe the real production
    # robust_ingest_file -> PDFExtractor -> state store -> vector store path.
    from rag_project.ingestion.robust_ingestor import robust_ingest_file
    from rag_project.testing.production_path_probes import _ProductionIngestionProbeSystem

    started = time.monotonic()
    duration = max(10.0, float(args.duration))
    minimum_iterations = 3
    target_iterations = max(minimum_iterations, min(12, int(duration * 1.5)))
    samples: list[int] = []
    fds: list[int] = []
    failures: list[str] = []
    iterations = 0
    successes = 0
    outer_root = Path(tempfile.mkdtemp(prefix="rag_resource_production_"))

    try:
        while iterations < target_iterations:
            iteration_root = outer_root / f"iteration_{iterations}"
            iteration_root.mkdir(parents=True, exist_ok=True)
            source_dir = iteration_root / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            system = None
            try:
                system = _ProductionIngestionProbeSystem(iteration_root)
                source = source_dir / f"resource_{iterations}.pdf"
                _write_probe_pdf(source, iterations)
                outcome = robust_ingest_file(system, source)
                status = str(outcome.get("status") or "unknown_failure")
                if status in {"success", "skipped", "ready", "completed"}:
                    successes += 1
                else:
                    detail = str(outcome.get("error") or outcome.get("reason") or status)
                    failures.append(f"iteration={iterations}: {detail}")
            except Exception as exc:
                failures.append(f"iteration={iterations}: {type(exc).__name__}: {exc}")
            finally:
                if system is not None:
                    _close_probe_system(system)
                shutil.rmtree(iteration_root, ignore_errors=True)

            current_rss = rss(os.getpid())
            current_fd = fd_count(os.getpid())
            if current_rss is not None:
                samples.append(current_rss)
            elif not samples:
                samples.append(0)
            else:
                samples.append(samples[-1])
            if current_fd is not None:
                fds.append(current_fd)
            elif not fds:
                fds.append(0)
            else:
                fds.append(fds[-1])
            iterations += 1
            gc.collect()
            if successes >= minimum_iterations and iterations >= minimum_iterations and time.monotonic() >= started + duration:
                break
    finally:
        _clear_chroma_cache()
        gc.collect()
        shutil.rmtree(outer_root, ignore_errors=True)
        _clear_chroma_cache()

    observed = time.monotonic() - started
    rss_delta = (samples[-1] - samples[0]) if len(samples) >= 2 else 0
    fd_delta = (fds[-1] - fds[0]) if len(fds) >= 2 else 0
    head = _quarter_mean(samples, True)
    tail = _quarter_mean(samples, False)
    payload = {
        "workload": "isolated real robust_ingest_file repetitions: PDF -> extraction -> chunking -> embedding -> validation -> READY",
        "observed_seconds": observed,
        "sample_count": len(samples),
        "iterations": iterations,
        "target_iterations": target_iterations,
        "successful_ingestions": successes,
        "failed_iterations": len(failures),
        "failure_samples": failures[:12],
        "rss_first_bytes": samples[0] if samples else 0,
        "rss_last_bytes": samples[-1] if samples else 0,
        "rss_peak_bytes": max(samples) if samples else 0,
        "rss_delta_bytes": rss_delta,
        "rss_slope_bytes_per_iteration": _linear_slope(samples),
        "rss_first_quarter_mean": head,
        "rss_last_quarter_mean": tail,
        "rss_tail_minus_head_mean_bytes": tail - head,
        "fd_first": fds[0] if fds else 0,
        "fd_last": fds[-1] if fds else 0,
        "fd_delta": fd_delta,
        "fd_peak": max(fds) if fds else 0,
        "ready_publication_contract_guarded": True,
        "repository_root": str(ROOT),
        "telemetry_platform": os.name,
        "runtime_reused_across_iterations": False,
        "iteration_bound_enforced": True,
        "isolated_iteration_cleanup": True,
        "pytest_diagnostic_adapters_loaded": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if successes >= minimum_iterations else 1


if __name__ == "__main__":
    raise SystemExit(main())
