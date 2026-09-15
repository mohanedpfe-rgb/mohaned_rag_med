from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import rag_project.storage.atomic_index_transaction as atomic


class _FailingCollection:
    def get(self, **kwargs):
        raise RuntimeError("preflight unavailable")

    def delete(self, **kwargs):
        raise AssertionError("compensation must not be able to mask the root failure")


class _Store:
    collection = _FailingCollection()
    _calls = 0

    def __init__(self, database: Path):
        self.lexical_database = database

    def _upsert_lexical_records(self, documents, metadatas, ids):
        raise AssertionError("not reached when preflight fails")

    def _lexical_tokens(self, document):
        return document.split()



def test_atomic_add_fails_closed_when_preflight_cannot_establish_compensation_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(atomic, "_INSTALLED", False)
    monkeypatch.setattr("rag_project.storage.vector_store.VectorStore", _Store)

    atomic.install()

    store = _Store(tmp_path / "lexical.sqlite3")
    with pytest.raises(RuntimeError, match="preflight unavailable"):
        store.add_documents(
            ["document"],
            [{"document_id": "doc", "chunk_id": "chunk"}],
            [[1.0, 0.0]],
            ["chunk"],
        )

    assert not (tmp_path / "lexical.sqlite3").exists() or _sqlite_row_count(tmp_path / "lexical.sqlite3") == 0


def _sqlite_row_count(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS lexical_documents ("
            "id TEXT PRIMARY KEY, document TEXT NOT NULL, metadata TEXT NOT NULL, "
            "index_state TEXT NOT NULL, tokens TEXT NOT NULL)"
        )
        row = connection.execute("SELECT COUNT(*) FROM lexical_documents").fetchone()
    return int(row[0] if row else 0)
