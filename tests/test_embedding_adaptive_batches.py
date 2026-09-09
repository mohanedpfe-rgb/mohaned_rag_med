from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.storage.vector_store import VectorStore


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _vectors(count: int, dimension: int = 4):
    return [[0.1 + (i / 100.0)] * dimension for i in range(count)]


def test_ollama_batch_splits_on_http_413(monkeypatch):
    service = EmbeddingService("http://127.0.0.1:11434", "nomic-embed-text", batch_size=16, retries=0)
    service._check_ollama_available = lambda force=False: True
    calls = []

    def fake_post(_url, json, **_kwargs):
        calls.append(len(json["input"]))
        if len(json["input"]) > 2:
            return _FakeResponse(413, {"error": "request too large"})
        return _FakeResponse(200, {"embeddings": _vectors(len(json["input"]))})

    monkeypatch.setattr("rag_project.embeddings.embedding_service.requests.post", fake_post)
    result = service.embed_texts(["a", "b", "c", "d"])

    assert len(result) == 4
    assert calls[0] == 4
    assert sorted(calls[1:]) == [2, 2]


def test_ollama_batch_splits_after_timeout(monkeypatch):
    service = EmbeddingService("http://127.0.0.1:11434", "nomic-embed-text", batch_size=8, retries=0)
    service._check_ollama_available = lambda force=False: True
    calls = []

    def fake_post(_url, json, **_kwargs):
        calls.append(len(json["input"]))
        if len(json["input"]) > 2:
            raise __import__("requests").exceptions.Timeout("simulated timeout")
        return _FakeResponse(200, {"embeddings": _vectors(len(json["input"]))})

    monkeypatch.setattr("rag_project.embeddings.embedding_service.requests.post", fake_post)
    result = service.embed_texts(["a", "b", "c", "d"])

    assert len(result) == 4
    assert calls[0] == 4
    assert sorted(calls[1:]) == [2, 2]


def test_settings_from_app_clamp_upgrades_legacy_tiny_batch(monkeypatch):
    from app import _clamp_local_embedding_profile

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("EMBEDDING_RETRIES", "0")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "20")
    _clamp_local_embedding_profile()

    assert int(__import__("os").environ["EMBEDDING_BATCH_SIZE"]) == 16
    assert int(__import__("os").environ["EMBEDDING_RETRIES"]) == 1
    assert float(__import__("os").environ["EMBEDDING_TIMEOUT_SECONDS"]) == 30.0


def test_vector_validation_preserves_numpy_embedding_arrays():
    store = VectorStore.__new__(VectorStore)
    store.collection = SimpleNamespace(
        metadata={"dimension": 4},
        get=lambda **_kwargs: {
            "ids": np.array(["chunk-1", "chunk-2"]),
            "documents": np.array(["a", "b"]),
            "metadatas": np.array([
                {"document_id": "doc", "chunk_id": "chunk-1", "version_id": "v", "index_state": "BUILDING"},
                {"document_id": "doc", "chunk_id": "chunk-2", "version_id": "v", "index_state": "BUILDING"},
            ], dtype=object),
            "embeddings": np.array(_vectors(2), dtype=float),
        },
    )
    result = store.validate_document_index("doc", "v")
    assert result["valid"] is True
    assert result["count"] == 2
