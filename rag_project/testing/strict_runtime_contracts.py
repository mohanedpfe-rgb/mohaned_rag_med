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


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None or isinstance(value, bool): raise RuntimeError(f"resource telemetry field missing: {key}")
    try: return float(value)
    except (TypeError, ValueError) as exc: raise RuntimeError(f"resource telemetry field invalid: {key}={value!r}") from exc


def strict_resource_stability(phase: Any) -> PhaseResult:
    result = PhaseResult(phase.number, phase.key, phase.name, status="FAIL", started_at=time.time())
    try:
        child = ROOT / "scripts" / "diagnostic_resource_workload.py"
        requested_mode = os.getenv("DIAGNOSTIC_RESOURCE_MODE", "bounded").strip().lower()
        requested_seconds = float(os.getenv("DIAGNOSTIC_RESOURCE_SECONDS", "5"))
        if requested_mode not in {"bounded", "24h"}: raise RuntimeError(f"unsupported resource certification mode: {requested_mode}")
        duration = max(requested_seconds, 86400.0) if requested_mode == "24h" else max(5.0, requested_seconds)
        proc = subprocess.run([sys.executable, str(child), "--duration", str(duration)], cwd=ROOT, text=True, capture_output=True, timeout=max(30, int(duration) + 30))
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            try: candidate = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(candidate, dict) and "observed_seconds" in candidate: payload = candidate; break
        if payload is None: raise RuntimeError(f"resource workload emitted no JSON payload; stdout_tail={proc.stdout[-1500:]}; stderr_tail={proc.stderr[-1500:]}")
        observed_seconds = _required_number(payload, "observed_seconds"); rss_delta = _required_number(payload, "rss_delta_bytes"); fd_delta = _required_number(payload, "fd_delta"); tail_head = _required_number(payload, "rss_tail_minus_head_mean_bytes"); slope = _required_number(payload, "rss_slope_bytes_per_iteration")
        samples = int(_required_number(payload, "sample_count")); iterations = int(_required_number(payload, "iterations")); successful = int(_required_number(payload, "successful_ingestions"))
        if observed_seconds <= 0 or samples < 3 or iterations < 3 or successful < 3: raise RuntimeError("resource observation depth is insufficient")
        max_reasonable_slope = max(256 * 1024, 64 * 1024 * 1024 / max(iterations, 1)); trend_ok = tail_head <= 32 * 1024 * 1024 and slope <= max_reasonable_slope; fd_ok = abs(int(fd_delta)) <= 2; rss_delta_ok = rss_delta <= 64 * 1024 * 1024
        result.details = {"evidence_level":"real_subprocess_resource_observation","evidence_level_extended":"strict_telemetry_required_trend_aware_subprocess_resource_observation","requested_seconds":duration,"requested_mode":requested_mode,"observed_seconds":observed_seconds,"sample_count":samples,"iterations":iterations,"repetitions":iterations,"successful_ingestions":successful,"rss_first_bytes":payload.get("rss_first_bytes"),"rss_last_bytes":payload.get("rss_last_bytes"),"rss_peak_bytes":payload.get("rss_peak_bytes"),"rss_delta_bytes":rss_delta,"rss_slope_bytes_per_iteration":slope,"rss_first_quarter_mean":payload.get("rss_first_quarter_mean"),"rss_last_quarter_mean":payload.get("rss_last_quarter_mean"),"rss_tail_minus_head_mean_bytes":tail_head,"fd_first":payload.get("fd_first"),"fd_last":payload.get("fd_last"),"fd_delta":fd_delta,"fd_peak":payload.get("fd_peak"),"trend_budget_bytes_per_iteration":max_reasonable_slope,"rss_trend_ok":trend_ok,"rss_delta_ok":rss_delta_ok,"fd_leak_ok":fd_ok,"observation_depth_ok":True,"telemetry_complete":True,"exit_code":proc.returncode,"workload":payload.get("workload"),"pipeline_exercised":["robust_ingest_file","PDFExtractor","SemanticChunker","EmbeddingService(test_mode)","VectorStore","IngestionStateStore","RSS trend sampling","FD sampling"],"long_running_24h_mode_supported":True,"certification_mode":"24h_observation" if requested_mode == "24h" else "bounded_smoke"}
        passed = proc.returncode == 0 and trend_ok and rss_delta_ok and fd_ok; result.score = 1.0 if passed else 0.0; result.status = "PASS" if passed else "FAIL"
        if not passed: result.failures.append({"location":"phase 15 strict telemetry/resource contract","exception":"ResourceStabilityContractFailure","message":str(result.details)})
    except Exception as exc:
        result.status="FAIL"; result.score=0.0; result.failures.append({"location":"phase 15 strict telemetry/resource contract","exception":type(exc).__name__,"message":str(exc)})
    result.duration_s=round(time.time()-result.started_at,3); return result


def _strict_semantic_wrapper(original):
    def wrapped(results): return list(original(results))
    return wrapped


def install() -> None:
    from rag_project.testing import runner
    runner.phase15_resource_stability = strict_resource_stability
    runner.UnifiedDiagnosticEngine._execute.__globals__["phase15_resource_stability"] = strict_resource_stability


__all__ = ["strict_resource_stability", "install"]
