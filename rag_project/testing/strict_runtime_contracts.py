from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]


def strict_resource_stability(phase: Any) -> PhaseResult:
    result = PhaseResult(phase.number, phase.key, phase.name, status="FAIL", started_at=time.time())
    try:
        child = ROOT / "scripts" / "diagnostic_resource_workload.py"
        requested_mode = os.getenv("DIAGNOSTIC_RESOURCE_MODE", "bounded").strip().lower()
        requested_seconds = float(os.getenv("DIAGNOSTIC_RESOURCE_SECONDS", "20"))
        duration = max(requested_seconds, 86400.0) if requested_mode == "24h" else requested_seconds
        proc = subprocess.run([sys.executable, str(child), "--duration", str(max(5.0, duration))], cwd=ROOT, text=True, capture_output=True, timeout=max(30, int(duration) + 30))
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "observed_seconds" in candidate:
                payload = candidate
                break
        if payload is None:
            raise RuntimeError(f"resource workload emitted no JSON payload; stdout_tail={proc.stdout[-1500:]}")
        rss_delta = payload.get("rss_delta_bytes")
        fd_delta = payload.get("fd_delta")
        tail_head = payload.get("rss_tail_minus_head_mean_bytes")
        slope = payload.get("rss_slope_bytes_per_iteration")
        samples = int(payload.get("sample_count") or 0)
        iterations = int(payload.get("iterations") or 0)
        successful = int(payload.get("successful_ingestions") or 0)
        max_reasonable_slope = max(256 * 1024, 64 * 1024 * 1024 / max(iterations, 1))
        trend_ok = (tail_head is None or float(tail_head) <= 32 * 1024 * 1024) and (slope is None or float(slope) <= max_reasonable_slope)
        fd_ok = fd_delta is None or abs(int(fd_delta)) <= 2
        rss_delta_ok = rss_delta is None or float(rss_delta) <= 64 * 1024 * 1024
        enough_observation = samples >= 3 and iterations >= 3 and successful >= 3
        result.details = {
            "evidence_level": "trend_aware_subprocess_resource_observation",
            "requested_seconds": duration,
            "requested_mode": requested_mode,
            "observed_seconds": payload.get("observed_seconds"),
            "sample_count": samples,
            "iterations": iterations,
            "repetitions": iterations,
            "successful_ingestions": successful,
            "rss_first_bytes": payload.get("rss_first_bytes"),
            "rss_last_bytes": payload.get("rss_last_bytes"),
            "rss_peak_bytes": payload.get("rss_peak_bytes"),
            "rss_delta_bytes": rss_delta,
            "rss_slope_bytes_per_iteration": slope,
            "rss_first_quarter_mean": payload.get("rss_first_quarter_mean"),
            "rss_last_quarter_mean": payload.get("rss_last_quarter_mean"),
            "rss_tail_minus_head_mean_bytes": tail_head,
            "fd_first": payload.get("fd_first"),
            "fd_last": payload.get("fd_last"),
            "fd_delta": fd_delta,
            "fd_peak": payload.get("fd_peak"),
            "trend_budget_bytes_per_iteration": max_reasonable_slope,
            "rss_trend_ok": trend_ok,
            "rss_delta_ok": rss_delta_ok,
            "fd_leak_ok": fd_ok,
            "observation_depth_ok": enough_observation,
            "exit_code": proc.returncode,
            "workload": payload.get("workload"),
            "pipeline_exercised": ["robust_ingest_file", "PDFExtractor", "SemanticChunker", "EmbeddingService(test_mode)", "VectorStore", "IngestionStateStore", "RSS trend sampling", "FD sampling"],
            "long_running_24h_mode_supported": True,
            "certification_mode": "24h_observation" if requested_mode == "24h" else "bounded_smoke",
        }
        passed = proc.returncode == 0 and enough_observation and trend_ok and rss_delta_ok and fd_ok
        result.score = 1.0 if passed else 0.0
        result.status = "PASS" if passed else "FAIL"
        if not passed:
            result.failures.append({"location": "phase 15 trend-aware resource contract", "exception": "ResourceStabilityContractFailure", "message": str(result.details)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 15 trend-aware resource contract", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def install() -> None:
    from rag_project.testing import runner
    runner.phase15_resource_stability = strict_resource_stability
    runner.UnifiedDiagnosticEngine._execute.__globals__["phase15_resource_stability"] = strict_resource_stability


__all__ = ["strict_resource_stability", "install"]
