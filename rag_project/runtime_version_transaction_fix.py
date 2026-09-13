from __future__ import annotations

import copy
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _snapshot(system: Any, source: Path) -> dict[str, Any] | None:
    state_store = getattr(system, "state_store", None)
    vector_store = getattr(system, "vector_store", None)
    if state_store is None or vector_store is None:
        return None

    previous = None
    try:
        previous = state_store.get_by_path(str(source.resolve()))
    except Exception:
        previous = None
    if not previous:
        return None

    document_id = str(previous.get("document_id") or "")
    if not document_id:
        return None

    snapshot: dict[str, Any] = {
        "document_id": document_id,
        "state": copy.deepcopy(previous),
        "file_bytes": source.read_bytes() if source.exists() else None,
        "vector": {},
        "lexical": [],
        "pages": [],
    }

    try:
        snapshot["vector"] = vector_store.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception:
        snapshot["vector"] = {}

    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            snapshot["lexical"] = connection.execute(
                "SELECT id, document, metadata, index_state, tokens "
                "FROM lexical_documents "
                "WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            ).fetchall()
    except Exception:
        snapshot["lexical"] = []

    try:
        snapshot["pages"] = state_store.get_pages(document_id)
    except Exception:
        snapshot["pages"] = []
    return snapshot


def _restore(system: Any, snapshot: dict[str, Any], source: Path) -> None:
    state_store = getattr(system, "state_store", None)
    vector_store = getattr(system, "vector_store", None)
    if state_store is None or vector_store is None:
        return

    document_id = snapshot["document_id"]
    vector = snapshot.get("vector") or {}
    ids = [str(value) for value in (vector.get("ids") or [])]
    documents = list(vector.get("documents") or [])
    metadatas = [dict(value or {}) for value in (vector.get("metadatas") or [])]
    embeddings = list(vector.get("embeddings") or [])
    if ids:
        try:
            kwargs: dict[str, Any] = {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
            }
            if embeddings and len(embeddings) == len(ids):
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

    previous_state = copy.deepcopy(snapshot.get("state") or {})
    if previous_state:
        previous_state["lease_owner"] = None
        previous_state["lease_expires_at"] = None
        previous_state["heartbeat_at"] = None
        try:
            state_store.upsert_document(previous_state)
        except Exception:
            pass

    try:
        state_store.delete_pages(document_id)
    except Exception:
        pass
    for page in snapshot.get("pages") or []:
        values = dict(page)
        values.pop("document_id", None)
        values.pop("page_number", None)
        try:
            state_store.upsert_page(
                document_id,
                int(page["page_number"]),
                **values,
            )
        except Exception:
            pass

    original_bytes = snapshot.get("file_bytes")
    if original_bytes is not None:
        try:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(original_bytes)
        except Exception:
            pass


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
            snapshot = None
            try:
                snapshot = _snapshot(self, source)
            except Exception:
                snapshot = None

            try:
                result = original(self, source, *args, **kwargs)
            except Exception:
                if snapshot is not None:
                    _restore(self, snapshot, source)
                raise

            failed = isinstance(result, dict) and str(result.get("status", "")).casefold() == "failed"
            if failed and snapshot is not None:
                _restore(self, snapshot, source)
            return result

        ingest_file._runtime_version_transaction_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", "ingest_file")
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
