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
        proc = subprocess.run(
            [sys.executable, str(child), "--duration", str(max(5.0, duration))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(30, int(duration) + 30),
        )
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
            "evidence_level": "real_subprocess_resource_observation",
            "evidence_level_extended": "trend_aware_subprocess_resource_observation",
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
            "pipeline_exercised": [
                "robust_ingest_file",
                "PDFExtractor",
                "SemanticChunker",
                "EmbeddingService(test_mode)",
                "VectorStore",
                "IngestionStateStore",
                "RSS trend sampling",
                "FD sampling",
            ],
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


def _with_provenance(original, phase, results):
    result = original(phase, results)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=True).stdout.strip()
        expected = os.getenv("GITHUB_SHA", "").strip()
        provenance_ok = bool(sha) and not status and (not expected or sha == expected)
        result.details["certification_provenance"] = {
            "git_head_sha": sha,
            "working_tree_clean": not bool(status),
            "expected_ci_sha": expected or None,
            "matches_expected_ci_sha": (not expected or sha == expected),
            "provenance_verified": provenance_ok,
        }
        if not provenance_ok:
            result.status = "FAIL"
            result.score = 0.0
            result.failures.append({"location": "phase 17 certification provenance", "exception": "CertificationProvenanceFailure", "message": str(result.details["certification_provenance"])})
    except Exception as exc:
        result.status = "FAIL"
        result.score = 0.0
        result.failures.append({"location": "phase 17 certification provenance", "exception": type(exc).__name__, "message": str(exc)})
    return result


def _strict_semantic_wrapper(original):
    def wrapped(results: dict[int, PhaseResult]):
        failures = list(original(results))
        extra: list[dict[str, Any]] = []

        try:
            from rag_project.testing.implementation_contracts import validate_runtime_ownership
            ownership = validate_runtime_ownership()
            if results.get(17) is not None:
                results[17].details["implementation_ownership_contract"] = ownership
            if not ownership.get("pass"):
                extra.append({"phase": 17, "reason": "authoritative 17-phase implementation ownership contract failed", "failures": ownership.get("failures", [])})
        except Exception as exc:
            extra.append({"phase": 17, "reason": "implementation ownership contract could not be evaluated", "exception": type(exc).__name__, "message": str(exc)})

        p2 = results.get(2)
        d2 = p2.details if p2 else {}
        if p2 is None or p2.status != "PASS" or d2.get("evidence_level") != "strict_fast_runtime_health" or not d2.get("checks", {}).get("production_import_smoke_exit_zero") or not d2.get("checks", {}).get("compileall_exit_zero"):
            extra.append({"phase": 2, "reason": "strict Phase 2 runtime health evidence missing"})
        p7 = results.get(7)
        d7 = p7.details if p7 else {}
        if p7 is None or int(d7.get("variant_count") or 0) < 5 or not d7.get("variant_results") or not all(bool(v) for v in d7.get("variant_results", {}).values()):
            extra.append({"phase": 7, "reason": "five-way adversarial PDF variant evidence incomplete"})
        p8 = results.get(8)
        d8 = p8.details if p8 else {}
        checks8 = d8.get("checks") or {}
        if p8 is None or len(checks8) < 8 or not all(bool(v) for v in checks8.values()) or not d8.get("ollama_protocol_path_executed"):
            extra.append({"phase": 8, "reason": "end-to-end metamorphic matrix incomplete"})
        p10 = results.get(10)
        d10 = p10.details if p10 else {}
        if p10 is None or d10.get("generation_client") != "OllamaLLMClient" or not d10.get("ollama_protocol_roundtrip_verified"):
            extra.append({"phase": 10, "reason": "production Ollama client roundtrip evidence incomplete"})
        p15 = results.get(15)
        d15 = p15.details if p15 else {}
        if p15 is None or not d15.get("rss_trend_ok") or not d15.get("fd_leak_ok") or d15.get("evidence_level") != "real_subprocess_resource_observation":
            extra.append({"phase": 15, "reason": "trend-aware resource evidence incomplete"})
        p16 = results.get(16)
        d16 = p16.details if p16 else {}
        if p16 is None or float(d16.get("evidence_grounding_case_rate") or 0) < 0.8 or float(d16.get("evidence_term_recall") or 0) < 0.8:
            extra.append({"phase": 16, "reason": "evidence-grounding benchmark below threshold"})
        return failures + extra
    return wrapped


def install() -> None:
    from rag_project.testing import runner, production_diagnostic_probes
    from rag_project.testing.full_metamorphic_probes import run_full_metamorphic_suite
    from rag_project.testing.full_mutation_probes import run_full_mutation_suite
    runner.phase15_resource_stability = strict_resource_stability
    runner.UnifiedDiagnosticEngine._execute.__globals__["phase15_resource_stability"] = strict_resource_stability
    runner.UnifiedDiagnosticEngine._execute.__globals__["metamorphic"] = run_full_metamorphic_suite
    runner.UnifiedDiagnosticEngine._execute.__globals__["_hardened_mutation_phase"] = run_full_mutation_suite
    runner._hardened_mutation_phase = run_full_mutation_suite
    original_phase17 = runner.UnifiedDiagnosticEngine._execute.__globals__.get("phase17_strict_completion")
    if original_phase17 is not None and not getattr(original_phase17, "_provenance_wrapped", False):
        def wrapped_phase17(phase, results):
            return _with_provenance(original_phase17, phase, results)
        wrapped_phase17._provenance_wrapped = True
        runner.UnifiedDiagnosticEngine._execute.__globals__["phase17_strict_completion"] = wrapped_phase17
    if not getattr(production_diagnostic_probes._semantic_contracts, "_strict_wrapped", False):
        production_diagnostic_probes._semantic_contracts = _strict_semantic_wrapper(production_diagnostic_probes._semantic_contracts)
        production_diagnostic_probes._semantic_contracts._strict_wrapped = True


__all__ = ["strict_resource_stability", "install"]
