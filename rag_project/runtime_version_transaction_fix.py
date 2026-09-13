from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _purge_unpublished_replacement(system: Any, snapshot: dict[str, Any]) -> None:
    """Remove every vector/lexical record for the document that was not in the READY snapshot."""
    vector_store = getattr(system, "vector_store", None)
    if vector_store is None:
        return
    document_id = str(snapshot.get("document_id") or "")
    if not document_id:
        return

    protected_ids = {
        str(value)
        for value in ((snapshot.get("vector") or {}).get("ids") or [])
        if str(value)
    }
    try:
        records = vector_store.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        current_ids = [str(value) for value in (records.get("ids") or [])]
        stale_ids = [value for value in current_ids if value not in protected_ids]
        if stale_ids:
            vector_store.collection.delete(ids=stale_ids)
    except Exception:
        pass

    protected_lexical = {
        str(row[0])
        for row in (snapshot.get("lexical") or [])
        if row and row[0] is not None
    }
    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            current = connection.execute(
                "SELECT id FROM lexical_documents "
                "WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            ).fetchall()
            stale_lexical = [str(row[0]) for row in current if str(row[0]) not in protected_lexical]
            if stale_lexical:
                connection.executemany(
                    "DELETE FROM lexical_documents WHERE id = ?",
                    [(value,) for value in stale_lexical],
                )
                connection.commit()
    except Exception:
        pass


def _restore_snapshot(system: Any, snapshot: dict[str, Any]) -> None:
    vector_store = getattr(system, "vector_store", None)
    state_store = getattr(system, "state_store", None)
    if vector_store is None or state_store is None:
        return

    _purge_unpublished_replacement(system, snapshot)

    vector = snapshot.get("vector") or {}
    ids = [str(value) for value in (vector.get("ids") or [])]
    if ids:
        try:
            kwargs: dict[str, Any] = {
                "ids": ids,
                "documents": list(vector.get("documents") or []),
                "metadatas": [dict(value or {}) for value in (vector.get("metadatas") or [])],
            }
            embeddings = list(vector.get("embeddings") or [])
            if len(embeddings) == len(ids):
                kwargs["embeddings"] = embeddings
            vector_store.collection.upsert(**kwargs)
        except Exception:
            pass

    rows = snapshot.get("lexical") or []
    if rows:
        try:
            with sqlite3.connect(vector_store.lexical_database) as connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO lexical_documents "
                    "(id, document, metadata, index_state, tokens) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                connection.commit()
        except Exception:
            pass

    previous_state = dict(snapshot.get("state") or {})
    if previous_state:
        previous_state["lease_owner"] = None
        previous_state["lease_expires_at"] = None
        previous_state["heartbeat_at"] = None
        try:
            state_store.upsert_document(previous_state)
        except Exception:
            pass

    document_id = str(snapshot.get("document_id") or "")
    try:
        state_store.delete_pages(document_id)
    except Exception:
        pass
    for page in snapshot.get("pages") or []:
        values = dict(page)
        values.pop("document_id", None)
        values.pop("page_number", None)
        try:
            state_store.upsert_page(document_id, int(page["page_number"]), **values)
        except Exception:
            pass


def _capture_snapshot(system: Any, source: Path) -> dict[str, Any] | None:
    try:
        state_store = getattr(system, "state_store", None)
        vector_store = getattr(system, "vector_store", None)
        if state_store is None or vector_store is None or not source.exists():
            return None
        previous = state_store.get_by_path(str(source.resolve()))
        if not previous:
            return None
        document_id = str(previous.get("document_id") or "")
        if not document_id:
            return None
        vector = vector_store.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
        with sqlite3.connect(vector_store.lexical_database) as connection:
            lexical = connection.execute(
                "SELECT id, document, metadata, index_state, tokens "
                "FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            ).fetchall()
        return {
            "document_id": document_id,
            "state": dict(previous),
            "vector": vector,
            "lexical": lexical,
            "pages": state_store.get_pages(document_id),
        }
    except Exception:
        return None


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.app.rag_system import RAGSystem

        original = getattr(RAGSystem, "ingest_file", None)
        if not callable(original) or getattr(original, "_runtime_version_transaction_fix", False):
            _INSTALLED = True
            return

        def ingest_file(self: Any, pdf_path: str | Path, *args: Any, **kwargs: Any):
            source = Path(pdf_path)
            snapshot = _capture_snapshot(self, source)
            try:
                result = original(self, source, *args, **kwargs)
            except Exception:
                if snapshot is not None:
                    _restore_snapshot(self, snapshot)
                raise

            failed = isinstance(result, dict) and str(result.get("status") or "").casefold() == "failed"
            if failed and snapshot is not None:
                _restore_snapshot(self, snapshot)
            return result

        ingest_file._runtime_version_transaction_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", ingest_file.__name__)
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
