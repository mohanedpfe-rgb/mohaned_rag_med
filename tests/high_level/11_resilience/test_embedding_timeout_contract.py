from __future__ import annotations

import pytest

from rag_project.embeddings.embedding_service import EmbeddingService


@pytest.mark.high_level
def test_resilience__embedding_timeout_configuration_is_bounded_to_safe_runtime_ceiling():
    service = EmbeddingService(
        "http://127.0.0.1:11434",
        "test-embedding",
        timeout_seconds=9999,
        test_mode=True,
    )
    assert service.timeout_seconds == 300.0


@pytest.mark.high_level
def test_resilience__embedding_timeout_configuration_has_safe_lower_bound():
    service = EmbeddingService(
        "http://127.0.0.1:11434",
        "test-embedding",
        timeout_seconds=0.01,
        test_mode=True,
    )
    assert service.timeout_seconds == 30.0


@pytest.mark.high_level
def test_resilience__test_mode_embedding_is_deterministic_and_does_not_contact_ollama(monkeypatch):
    service = EmbeddingService(
        "http://127.0.0.1:11434",
        "test-embedding",
        test_mode=True,
    )

    def forbidden_network(*args, **kwargs):
        raise AssertionError("test-mode embedding must not contact Ollama")

    monkeypatch.setattr("requests.post", forbidden_network)
    vectors = service.embed_texts(["controlled medical evidence", "controlled medical evidence"])

    assert len(vectors) == 2
    assert vectors[0] == vectors[1]
    assert service.last_error is None
