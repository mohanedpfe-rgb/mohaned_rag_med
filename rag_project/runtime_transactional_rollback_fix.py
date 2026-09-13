from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _capture_previous(system: Any, pdf_path: str | Path) -> dict[str, Any] | None:
    try:
        source = Path(pdf_path)
        if not source.is_file():
            return None
        content_hash = str(system._hash_file(source))
        previous = system.state_store.get_by_path(str(source.resolve()))
        if not previous or str(previous.get("content_hash") or "") == content_hash:
            return None
        document_id = str(previous.get("document_id") or "")
        if not document_id:
            return None

        records = system.vector_store.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
        lexical_rows: list[tuple[Any, ...]] = []
        with sqlite3.connect(system.vector_store.lexical_database) as connection:
            lexical_rows = connection.execute(
                "SELECT id, document, metadata, index_state, tokens "
                "FROM lexical_documents "
                "WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            ).fetchall()
        return {
            "document_id": document_id,
            "state": dict(previous),
            "ids": [str(value) for value in (records.get("ids") or [])],
            "documents": list(records.get("documents") or []),
            "metadatas": [dict(value or {}) for value in (records.get("metadatas") or [])],
            "embeddings": list(records.get("embeddings") or []),
            "lexical_rows": lexical_rows,
        }
    except Exception:
        return None


def _restore_previous(system: Any, snapshot: dict[str, Any]) -> None:
    ids = snapshot.get("ids") or []
    if ids:
        system.vector_store.collection.upsert(
            ids=[str(value) for value in ids],
            documents=[str(value) for value in snapshot.get("documents") or []],
            metadatas=[dict(value or {}) for value in snapshot.get("metadatas") or []],
            embeddings=[list(map(float, value)) for value in snapshot.get("embeddings") or []],
        )

    rows = snapshot.get("lexical_rows") or []
    if rows:
        with sqlite3.connect(system.vector_store.lexical_database) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO lexical_documents "
                "(id, document, metadata, index_state, tokens) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            connection.commit()

    previous_state = dict(snapshot.get("state") or {})
    if previous_state:
        # Restore only authoritative document-state fields. Lease ownership is
        # intentionally cleared so the failed replacement cannot keep the old
        # document permanently locked.
        previous_state.pop("lease_owner", None)
        previous_state.pop("lease_expires_at", None)
        previous_state.pop("heartbeat_at", None)
        try:
            system.state_store.upsert_document(previous_state)
        except Exception:
            pass


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.app.rag_system import RAGSystem

        original = getattr(RAGSystem, "ingest_file", None)
        if not callable(original) or getattr(original, "_transactional_rollback_fix", False):
            _INSTALLED = True
            return

        def ingest_file(self: Any, pdf_path: str | Path, *args: Any, **kwargs: Any):
            snapshot = _capture_previous(self, pdf_path)
            result = original(self, pdf_path, *args, **kwargs)
            if snapshot and isinstance(result, dict) and str(result.get("status") or "").casefold() == "failed":
                try:
                    _restore_previous(self, snapshot)
                except Exception as exc:
                    try:
                        self.logger.exception("Transactional rollback restoration failed: %s", exc)
                    except Exception:
                        pass
            return result

        ingest_file._transactional_rollback_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", "ingest_file")
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
