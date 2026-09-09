from __future__ import annotations

import time
from typing import Any

from rag_project.embeddings.embedding_service import EmbeddingService


def test_query_cache_isolated_when_provider_changes() -> None:
    service = EmbeddingService(
        "http://127.0.0.1:11434",
        "test-model",
        cache_size=8,
        cache_ttl_seconds=60,
    )
    calls: list[str] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(service.provider)
        return [[1.0, 0.0] for _ in texts]

    service._embed_batch = fake_embed
    assert service.embed_query("same query") == [1.0, 0.0]
    service.provider = "sentence-transformers"
    assert service.embed_query("same query") == [1.0, 0.0]
    assert calls == ["ollama", "sentence-transformers"]


def test_embedding_cache_expires() -> None:
    service = EmbeddingService(
        "http://127.0.0.1:11434",
        "test-model",
        cache_size=8,
        cache_ttl_seconds=0.01,
    )
    calls = 0

    def fake_embed(texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return [[0.5, 0.5] for _ in texts]

    service._embed_batch = fake_embed
    assert service.embed_texts(["same text"]) == [[0.5, 0.5]]
    assert service.embed_texts(["same text"]) == [[0.5, 0.5]]
    assert calls == 1
    time.sleep(0.02)
    assert service.embed_texts(["same text"]) == [[0.5, 0.5]]
    assert calls == 2


def test_embedding_cache_does_not_mix_model_identity() -> None:
    first = EmbeddingService(
        "http://127.0.0.1:11434",
        "model-a",
        cache_size=8,
        cache_ttl_seconds=60,
    )
    second = EmbeddingService(
        "http://127.0.0.1:11434",
        "model-b",
        cache_size=8,
        cache_ttl_seconds=60,
    )
    first._embed_batch = lambda texts: [[1.0, 0.0] for _ in texts]
    second._embed_batch = lambda texts: [[0.0, 1.0] for _ in texts]
    assert first.embed_texts(["same text"]) == [[1.0, 0.0]]
    assert second.embed_texts(["same text"]) == [[0.0, 1.0]]
