from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded, assert_latency_under, assert_pipeline_authority


@pytest.mark.high_level
def test_e2e_failure_recovery__llm_outage_produces_exact_verified_extractive_recovery(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast
    before = len(getattr(clean_system.conversation_memory, "history", []) or [])

    result = clean_system.answer("Explain the mechanism and management of diabetes mellitus using the indexed evidence.")

    assert_exact_status(result, "SUCCESS_WITH_WARNINGS")
    assert_exact_path(result, "PATH_A_VERIFIED_FALLBACK")
    assert result.get("recovery", {}).get("attempted") is True
    assert result.get("recovery", {}).get("pipeline_error") == "RuntimeError"
    assert result.get("recovery", {}).get("grounded_extractive_fallback") is True
    assert result.get("citations")
    assert result.get("hits")
    assert result.get("query_trace", {}).get("generation", {}).get("status") == "extractive"
    assert result.get("phase_implementation", {}).get("degraded_to_recovery") is True
    assert_citations_valid(result)
    assert_grounded(result)
    assert_pipeline_authority(result)
    assert_latency_under(result, 8.0)

    after = len(getattr(clean_system.conversation_memory, "history", []) or [])
    assert after == before + 1
