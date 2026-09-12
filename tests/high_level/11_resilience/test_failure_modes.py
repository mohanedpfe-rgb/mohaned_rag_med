from __future__ import annotations

import pytest

from rag_project.generation.llm_client import OllamaLLMClient
from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_one_of_statuses


@pytest.mark.high_level
@pytest.mark.multi_outcome
def test_resilience__llm_failure_produces_grounded_recovery_or_explicit_safe_abstention_without_crash(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management of diabetes mellitus using the indexed evidence."
    )

    status = str(result.get("status") or "").upper()
    assert_one_of_statuses(result, {"SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN"})
    assert "Traceback" not in str(result.get("answer") or "")

    recovery = result.get("recovery") or {}
    assert recovery.get("attempted") is True
    assert recovery.get("pipeline_error") == "RuntimeError"

    if status == "SUCCESS_WITH_WARNINGS":
        assert recovery.get("grounded_extractive_fallback") is True
        assert recovery.get("verification") in {
            "exact_extractive_provenance",
            "semantic_claim_verification",
        }
        assert result.get("hits")
        assert result.get("citations")
        assert result.get("grounding", {}).get("allow") is True
        assert result.get("phase_implementation", {}).get("degraded_to_recovery") is True
        assert result.get("query_trace", {}).get("generation", {}).get("status") == "extractive"
        assert_citations_valid(result)
        assert_grounded(result)
    else:
        assert result.get("citations") == []
        assert result.get("hits") in (None, [])


@pytest.mark.high_level
def test_resilience__ollama_circuit_breaker_opens_after_two_failures_and_short_circuits_third_request(monkeypatch):
    client = OllamaLLMClient(
        "http://127.0.0.1:1",
        "test-model",
        timeout_seconds=5,
        circuit_threshold=2,
        circuit_open_seconds=30,
    )

    def always_fail(payload):
        raise RuntimeError("simulated transport failure")

    monkeypatch.setattr(client, "_post_chat", always_fail)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client.generate("test")

    assert client._circuit_is_open() is True
    with pytest.raises(RuntimeError, match="circuit breaker is open"):
        client.generate("third request must be short-circuited")
