from __future__ import annotations

import rag_project.runtime_chroma_lifecycle_fix as lifecycle


class _Client:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _Store:
    def __init__(self):
        self.client = _Client()
        self.collection = object()
        self.expected_identity = object()
        self._chroma_closed = False


def test_chroma_lifecycle_close_is_idempotent(monkeypatch):
    monkeypatch.setattr(lifecycle, "_INSTALLED", False)
    monkeypatch.setattr("rag_project.storage.vector_store.VectorStore", _Store)

    lifecycle.install()

    store = _Store()
    client = store.client
    store.close()
    store.close()

    assert client.close_calls == 1
    assert store.client is None
    assert store.collection is None
    assert store.expected_identity is None
    assert store._chroma_closed is True
