from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded, assert_pipeline_authority


@pytest.mark.high_level
def test_resilience__ollama_failure_uses_verified_extractiver_recovery_without_uncited_success(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management of diabetes mellitus using the indexed evidence."
    )

    assert_exact_status(result, "SUCCESS_WITH_WARNINGS")
    assert_exact_path(result, "PATH_A_VERIFIED_FALLBACK")
    assert result.get("hits")
    assert result.get("citations")
    assert result.get("recovery", {}).get("attempted") is True
    assert result.get("recovery", {}).get("grounded_extractive_fallback") is True
    assert result.get("recovery", {}).get("pipeline_error") == "RuntimeError"
    assert_pipeline_authority(result)
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_resilience__successful_requests_retain_runtime_safety_observability(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert_pipeline_authority(result)

    safety = result.get("runtime_safety") or {}
    assert safety.get("ready_evidence_enforced") is True
    assert safety.get("success_contract_enforced") is True
