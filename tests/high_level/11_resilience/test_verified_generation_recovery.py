from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_resilience__ollama_outage_recovers_to_exact_verified_extractive_answer(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management of diabetes mellitus using the indexed evidence."
    )

    assert_exact_status(result, "SUCCESS_WITH_WARNINGS")
    assert_exact_path(result, "PATH_A_VERIFIED_FALLBACK")
    recovery = result.get("recovery") or {}
    assert recovery.get("attempted") is True
    assert recovery.get("pipeline_error") == "RuntimeError"
    assert recovery.get("grounded_extractive_fallback") is True
    assert result.get("hits")
    assert_citations_valid(result)
    assert_grounded(result)
    assert result.get("phase_implementation", {}).get("degraded_to_recovery") is True
    assert (result.get("query_trace") or {}).get("generation", {}).get("status") == "extractive"
