from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import rag_project.storage.vector_store_runtime as runtime


class _Collection:
    def __init__(self):
        self.updates: list[tuple[str, str]] = []

    def update(self, ids, metadatas):
        self.updates.append((str(ids[0]), str(metadatas[0]["index_state"])))


class _FailingConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def executemany(self, *args, **kwargs):
        raise RuntimeError("lexical state update failed")

    def commit(self):
        raise AssertionError("commit must not be reached after lexical update failure")


def test_set_version_index_state_restores_semantic_state_when_lexical_update_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    collection = _Collection()
    store = SimpleNamespace(collection=collection, lexical_database=tmp_path / "lexical.sqlite3")
    semantic = [
        ("chunk-1", {"chunk_id": "chunk-1", "index_state": "BUILDING"}),
        ("chunk-2", {"chunk_id": "chunk-2", "index_state": "BUILDING"}),
    ]
    lexical = [("chunk-1", {"chunk_id": "chunk-1"}), ("chunk-2", {"chunk_id": "chunk-2"})]

    monkeypatch.setattr(
        runtime,
        "_document_index_counts",
        lambda *_args, **_kwargs: {
            "semantic_count": 2,
            "lexical_count": 2,
            "semantic_chunk_ids": {"chunk-1", "chunk-2"},
            "lexical_chunk_ids": {"chunk-1", "chunk-2"},
            "valid": True,
        },
    )
    monkeypatch.setattr(runtime, "_semantic_records", lambda *_args, **_kwargs: semantic)
    monkeypatch.setattr(runtime, "_lexical_records", lambda *_args, **_kwargs: lexical)
    monkeypatch.setattr(runtime.sqlite3, "connect", lambda *_args, **_kwargs: _FailingConnection())

    with pytest.raises(RuntimeError, match="lexical state update failed"):
        runtime._set_version_index_state(store, "doc", "version", "READY")

    assert collection.updates == [
        ("chunk-1", "READY"),
        ("chunk-2", "READY"),
        ("chunk-2", "BUILDING"),
        ("chunk-1", "BUILDING"),
    ]
