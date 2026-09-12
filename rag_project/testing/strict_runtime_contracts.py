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
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"resource telemetry field missing: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"resource telemetry field invalid: {key}={value!r}") from exc


def strict_resource_stability(phase: Any) -> PhaseResult:
    result = PhaseResult(phase.number, phase.key, phase.name, status="FAIL", started_at=time.time())
    try:
        child = ROOT / "scripts" / "diagnostic_resource_workload.py"
        requested_mode = os.getenv("DIAGNOSTIC_RESOURCE_MODE", "bounded").strip().lower()
        requested_seconds = float(os.getenv("DIAGNOSTIC_RESOURCE_SECONDS", "20"))
        if requested_mode not in {"bounded", "24h"}:
            raise RuntimeError(f"unsupported resource certification mode: {requested_mode}")
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

        observed_seconds = _required_number(payload, "observed_seconds")
        rss_delta = _required_number(payload, "rss_delta_bytes")
        fd_delta = _required_number(payload, "fd_delta")
        tail_head = _required_number(payload, "rss_tail_minus_head_mean_bytes")
        slope = _required_number(payload, "rss_slope_bytes_per_iteration")
        samples = int(_required_number(payload, "sample_count"))
        iterations = int(_required_number(payload, "iterations"))
        successful = int(_required_number(payload, "successful_ingestions"))
        if observed_seconds <= 0 or samples < 3 or iterations < 3 or successful < 3:
            raise RuntimeError("resource observation depth is insufficient")

        max_reasonable_slope = max(256 * 1024, 64 * 1024 * 1024 / max(iterations, 1))
        trend_ok = tail_head <= 32 * 1024 * 1024 and slope <= max_reasonable_slope
        fd_ok = abs(int(fd_delta)) <= 2
        rss_delta_ok = rss_delta <= 64 * 1024 * 1024
        result.details = {
            "evidence_level": "real_subprocess_resource_observation",
            "evidence_level_extended": "strict_telemetry_required_trend_aware_subprocess_resource_observation",
            "requested_seconds": duration,
            "requested_mode": requested_mode,
            "observed_seconds": observed_seconds,
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
            "observation_depth_ok": True,
            "telemetry_complete": True,
            "exit_code": proc.returncode,
            "workload": payload.get("workload"),
            "pipeline_exercised": ["robust_ingest_file", "PDFExtractor", "SemanticChunker", "EmbeddingService(test_mode)", "VectorStore", "IngestionStateStore", "RSS trend sampling", "FD sampling"],
            "long_running_24h_mode_supported": True,
            "certification_mode": "24h_observation" if requested_mode == "24h" else "bounded_smoke",
        }
        passed = proc.returncode == 0 and trend_ok and rss_delta_ok and fd_ok
        result.score = 1.0 if passed else 0.0
        result.status = "PASS" if passed else "FAIL"
        if not passed:
            result.failures.append({"location": "phase 15 strict telemetry/resource contract", "exception": "ResourceStabilityContractFailure", "message": str(result.details)})
    except Exception as exc:
        result.status = "FAIL"
        result.score = 0.0
        result.failures.append({"location": "phase 15 strict telemetry/resource contract", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


def _strict_semantic_wrapper(original):
    def wrapped(results: dict[int, PhaseResult]):
        failures = list(original(results))
        extra: list[dict[str, Any]] = []
        try:
            from rag_project.testing.implementation_contracts import validate_runtime_ownership
            ownership = validate_runtime_ownership()
            if not ownership.get("pass"):
                extra.append({"phase": 17, "reason": "authoritative 17-phase implementation ownership contract failed", "failures": ownership.get("failures", [])})
        except Exception as exc:
            extra.append({"phase": 17, "reason": "implementation ownership contract could not be evaluated", "exception": type(exc).__name__, "message": str(exc)})

        for number, impl in ((3, "strict_diagnostic_chain"), (4, "strict_contract_triangulation"), (5, "strict_cross_layer_invariants"), (6, "strict_information_loss")):
            item = results.get(number); details = item.details if item else {}
            if item is None or details.get("authoritative_implementation") != impl or not details.get("fault_sensitivity_verified") or int(details.get("fault_probe_count") or 0) < 2:
                extra.append({"phase": number, "reason": f"fault-sensitive {impl} evidence incomplete"})

        p2 = results.get(2); d2 = p2.details if p2 else {}
        if p2 is None or p2.status != "PASS" or d2.get("evidence_level") != "strict_fast_runtime_health" or not d2.get("checks", {}).get("production_import_smoke_exit_zero") or not d2.get("checks", {}).get("compileall_exit_zero"):
            extra.append({"phase": 2, "reason": "strict Phase 2 runtime health evidence missing"})

        p7 = results.get(7); d7 = p7.details if p7 else {}
        if p7 is None or int(d7.get("variant_count") or 0) < 5 or not d7.get("variant_results") or not all(bool(v) for v in d7.get("variant_results", {}).values()):
            extra.append({"phase": 7, "reason": "five-way adversarial PDF variant evidence incomplete"})

        p8 = results.get(8); d8 = p8.details if p8 else {}; checks8 = d8.get("checks") or {}
        if p8 is None or len(checks8) < 8 or not all(bool(v) for v in checks8.values()) or not d8.get("ollama_protocol_path_executed"):
            extra.append({"phase": 8, "reason": "end-to-end metamorphic matrix incomplete"})

        p10 = results.get(10); d10 = p10.details if p10 else {}
        if p10 is None or d10.get("generation_client") != "OllamaLLMClient" or not d10.get("ollama_protocol_roundtrip_verified"):
            extra.append({"phase": 10, "reason": "production Ollama client roundtrip evidence incomplete"})

        p13 = results.get(13); d13 = p13.details if p13 else {}
        if p13 is None or d13.get("authoritative_implementation") != "strict_causal_phase" or not d13.get("causal_false_positive_control_verified"):
            extra.append({"phase": 13, "reason": "causal false-positive control incomplete"})

        p14 = results.get(14); d14 = p14.details if p14 else {}
        baseline_path = ROOT / "tests" / "support" / "performance_baseline.json"
        baseline_contract_ok = False
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            expected_stages = {"canonical_ingestion", "post_ingestion_validation", "lexical_retrieval", "semantic_retrieval"}
            baseline_contract_ok = baseline.get("schema_version") == 2 and baseline.get("baseline_type") == "certification_ceiling" and baseline.get("historical_measurement") is False and expected_stages.issubset(baseline.keys()) and baseline.get("canonical_stage_keys") == sorted(expected_stages) and all(isinstance(baseline.get(stage, {}).get("p95_ms"), (int, float)) for stage in expected_stages)
        except Exception:
            baseline_contract_ok = False
        if p14 is None or not d14.get("stage_metrics") or not d14.get("regression_comparisons") or not d14.get("regression_pass") or not baseline_contract_ok or not d14.get("benchmark_environment", {}).get("requirements_lock_sha256"):
            extra.append({"phase": 14, "reason": "performance benchmark/regression evidence or baseline/environment provenance incomplete"})
        else:
            d14["baseline_contract_verified"] = True; d14["baseline_type"] = "certification_ceiling"; d14["historical_baseline_available"] = False

        p15 = results.get(15); d15 = p15.details if p15 else {}
        if p15 is None or not d15.get("rss_trend_ok") or not d15.get("fd_leak_ok") or not d15.get("telemetry_complete") or d15.get("evidence_level") != "real_subprocess_resource_observation":
            extra.append({"phase": 15, "reason": "strict telemetry/trend-aware resource evidence incomplete"})

        p16 = results.get(16); d16 = p16.details if p16 else {}
        if p16 is None or d16.get("dataset_id") != "phase16_production_independent_v2" or not d16.get("gold_integrity_contract_verified") or not d16.get("gold_references_resolved") or not d16.get("independent_from_phase9_dataset") or float(d16.get("evidence_grounding_case_rate") or 0) < 0.8 or float(d16.get("evidence_term_recall") or 0) < 0.8:
            extra.append({"phase": 16, "reason": "strict independent Phase 16 corpus/gold contract incomplete"})

        return failures + extra
    return wrapped


def install() -> None:
    from rag_project.testing import runner, production_diagnostic_probes
    from rag_project.testing.full_metamorphic_probes import run_full_metamorphic_suite
    from rag_project.testing.full_mutation_probes import run_full_mutation_suite
    from rag_project.testing.strict_information_loss import strict_information_loss
    from rag_project.testing.strict_foundation_phases import strict_diagnostic_chain, strict_contract_triangulation, strict_cross_layer_invariants
    from rag_project.testing.strict_causal_phase import strict_causal_phase

    runner.phase15_resource_stability = strict_resource_stability
    globals_map = runner.UnifiedDiagnosticEngine._execute.__globals__
    globals_map["phase15_resource_stability"] = strict_resource_stability
    globals_map["metamorphic"] = run_full_metamorphic_suite
    globals_map["_hardened_mutation_phase"] = run_full_mutation_suite
    globals_map["information_loss"] = strict_information_loss
    globals_map["diagnostic_chain"] = strict_diagnostic_chain
    globals_map["contract_triangulation"] = strict_contract_triangulation
    globals_map["cross_layer_invariants"] = strict_cross_layer_invariants
    globals_map["phase13_known_causal_graph"] = strict_causal_phase
    runner._hardened_mutation_phase = run_full_mutation_suite

    if not getattr(production_diagnostic_probes._semantic_contracts, "_strict_wrapped", False):
        production_diagnostic_probes._semantic_contracts = _strict_semantic_wrapper(production_diagnostic_probes._semantic_contracts)
        production_diagnostic_probes._semantic_contracts._strict_wrapped = True


__all__ = ["strict_resource_stability", "install"]
