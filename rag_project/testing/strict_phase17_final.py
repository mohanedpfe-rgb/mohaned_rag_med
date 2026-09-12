"""Final authoritative Phase 17 certification boundary."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from rag_project.testing.deep_diagnostics import PhaseResult

ROOT = Path(__file__).resolve().parents[2]


def phase17_strict_completion(phase: Any, results: dict[int, PhaseResult]) -> PhaseResult:
    from rag_project.testing import runner
    from rag_project.testing.implementation_contracts import validate_runtime_ownership
    from rag_project.testing import production_diagnostic_probes

    result = PhaseResult(phase.number, phase.key, phase.name, status="FAIL", started_at=time.time())
    failures: list[dict[str, Any]] = []
    try:
        if phase.number != 17 or phase.key != "certification":
            failures.append({"phase": 17, "reason": "Phase 17 received a non-canonical PhaseSpec", "actual_number": phase.number, "actual_key": phase.key})

        identity_contract: dict[str, Any] = {}
        for number in range(1, 17):
            item = results.get(number)
            identity_contract[str(number)] = {"present": item is not None, "number": getattr(item, "number", None), "key": getattr(item, "key", None)}
            if item is None:
                failures.append({"phase": number, "reason": "completed phase result is missing before certification"})
            elif item.number != number:
                failures.append({"phase": number, "reason": "phase result identity mismatch", "observed_number": item.number})
            elif item.key != runner.PHASES[number - 1].key:
                failures.append({"phase": number, "reason": "phase result key mismatch", "observed_key": item.key, "expected_key": runner.PHASES[number - 1].key})

        semantic_failures = list(production_diagnostic_probes._semantic_contracts(results))
        failures.extend(semantic_failures)

        ownership = validate_runtime_ownership()
        if not ownership.get("pass"):
            failures.append({"phase": 17, "reason": "runtime implementation ownership contract failed", "ownership_failures": ownership.get("failures", [])})

        base = runner._base_phase17_strict(phase, results)
        failures.extend(base.details.get("evidence_failures", []))
        if base.status != "PASS":
            failures.extend(base.failures or [{"phase": 17, "reason": "base strict certification failed"}])

        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=True).stdout.strip()
        expected = os.getenv("GITHUB_SHA", "").strip()
        provenance_ok = bool(sha) and not status and (not expected or sha == expected)
        if not provenance_ok:
            failures.append({"phase": 17, "reason": "certification provenance failed", "git_head_sha": sha, "working_tree_clean": not bool(status), "expected_ci_sha": expected or None})

        result.details = {
            **(base.details or {}),
            "evidence_level": "strict_final_certification_boundary",
            "certification_mode": "validate_existing_phase_results_without_reexecution",
            "authoritative_recheck_phase_identity": {"phase_8_number": getattr(results.get(8), "number", None), "phase_11_number": getattr(results.get(11), "number", None), "verified": getattr(results.get(8), "number", None) == 8 and getattr(results.get(11), "number", None) == 11},
            "phase_result_identity_contract": identity_contract,
            "implementation_ownership_verified": ownership.get("pass", False),
            "certification_provenance": {"git_head_sha": sha, "working_tree_clean": not bool(status), "expected_ci_sha": expected or None, "matches_expected_ci_sha": not expected or sha == expected, "provenance_verified": provenance_ok},
            "semantic_contract_validation_executed": True,
            "reexecuted_phases": [],
            "authoritative_module": "rag_project.testing.strict_phase17_final",
            "legacy_phase17_wrapper_active": False,
        }
        result.details["evidence_failures"] = failures
        unique_failed = {int(x.get("phase", 17)) for x in failures if str(x.get("phase", "")).isdigit()}
        result.details["implementation_coverage"] = "17/17" if not failures else f"{17 - len(unique_failed)}/17"
        result.details["fully_implemented_phase_numbers"] = list(range(1, 18)) if not failures else []
        result.details["runtime_non_pass_phases"] = sorted(n for n, item in results.items() if item.status != "PASS")
        result.details["missing_phase_results"] = sorted(set(range(1, 17)) - set(results))
        result.details["phase_by_phase_evidence_contract"] = {"required_phases": list(range(1, 17)), "failures": failures}
        result.score = 1.0 if not failures else 0.0
        result.status = "PASS" if not failures else "FAIL"
        if failures:
            result.failures.append({"location": "phase 17 final certification boundary", "exception": "Incomplete17PhaseImplementation", "message": str(failures)})
    except Exception as exc:
        result.status = "FAIL"
        result.score = 0.0
        result.failures.append({"location": "phase 17 final certification boundary", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result


__all__ = ["phase17_strict_completion"]
