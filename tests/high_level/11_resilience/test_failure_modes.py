from __future__ import annotations

import pytest

from rag_project.generation.llm_client import OllamaLLMClient


@pytest.mark.high_level
def test_resilience__llm_failure_degrades_to_grounded_fallback_without_crashing(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management of diabetes mellitus using the indexed evidence."
    )

    assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE", "NOT_SUPPORTED"}
    assert "Traceback" not in str(result.get("answer") or "")
    if str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert result.get("hits")
        assert result.get("citations")
        assert result.get("verification", {}).get("allow") is True


@pytest.mark.high_level
def test_resilience__ollama_circuit_breaker_opens_after_threshold(monkeypatch):
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
