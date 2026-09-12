from __future__ import annotations

from rag_project.testing.full_mutation_probes import run_full_mutation_suite
from rag_project.testing.runner import PHASES


def test_authoritative_phase11_is_full_mutation_suite() -> None:
    result = run_full_mutation_suite(PHASES[10])
    assert result.status == "PASS", result.failures
    assert result.details["evidence_level"] == "executable_multi_module_mutation_suite"
    assert result.details["mutants_applicable"] >= 12
    assert result.details["mutants_killed"] == result.details["mutants_applicable"]
    assert result.details["kill_score"] == 1.0
    assert result.details["target_module_count"] >= 3
    assert result.details["real_pytest_subprocess"] is True
    assert result.details["checkout_modified"] is False
