from __future__ import annotations

import gc
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _current_document_records(vector_store: Any, document_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Read all Chroma records and filter by document identity in Python.

    Avoid relying on Chroma's ``where`` evaluator during rollback: the rollback
    contract must still work when an upgraded Chroma backend handles metadata
    filtering differently from the normal retrieval path.
    """
    records = vector_store.collection.get(include=["metadatas"])
    ids = [str(value) for value in (records.get("ids") or [])]
    metadata = [dict(value or {}) for value in (records.get("metadatas") or [])]
    matched_ids: list[str] = []
    matched_metadata: list[dict[str, Any]] = []
    for item_id, values in zip(ids, metadata, strict=False):
        if str(values.get("document_id") or values.get("doc_id") or "") == document_id:
            matched_ids.append(item_id)
            matched_metadata.append(values)
    return matched_ids, matched_metadata


def _purge_document_records(system: Any, document_id: str) -> None:
    """Atomically reduce the live index to zero records for one document.

    This is deliberately stronger than deleting a single version. A failed
    replacement must never leave a partially published vector or lexical row
    that remains discoverable after rollback.
    """
    vector_store = getattr(system, "vector_store", None)
    if vector_store is None or not document_id:
        return

    last_ids: list[str] = []
    for _attempt in range(3):
        try:
            ids, _metadata = _current_document_records(vector_store, document_id)
            last_ids = ids
            if ids:
                vector_store.collection.delete(ids=ids)
            verify_ids, _ = _current_document_records(vector_store, document_id)
            if not verify_ids:
                break
            # Some Chroma builds are more reliable when deletion is expressed
            # as a metadata predicate after an ID-based delete attempt.
            try:
                vector_store.collection.delete(where={"document_id": document_id})
            except Exception:
                pass
            verify_ids, _ = _current_document_records(vector_store, document_id)
            if not verify_ids:
                break
            last_ids = verify_ids
        except Exception:
            continue

    # Clear process-local GC references before the snapshot is re-published.
    gc.collect()
    if last_ids:
        raise RuntimeError(
            f"Rollback could not purge live vector records for document {document_id}: {last_ids}"
        )

    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            connection.execute(
                "DELETE FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (document_id,),
            )
            connection.commit()
    except Exception as exc:
        raise RuntimeError(
            f"Rollback could not purge lexical records for document {document_id}: {exc}"
        ) from exc


def _restore_snapshot(system: Any, snapshot: dict[str, Any]) -> None:
    vector_store = getattr(system, "vector_store", None)
    state_store = getattr(system, "state_store", None)
    if vector_store is None or state_store is None:
        return

    document_id = str(snapshot.get("document_id") or "")
    if not document_id:
        return

    # Fail closed: remove every current record for the document first, then
    # restore only the previously captured READY snapshot.
    _purge_document_records(system, document_id)

    vector = snapshot.get("vector") or {}
    ids = [str(value) for value in (vector.get("ids") or [])]
    if ids:
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": list(vector.get("documents") or []),
            "metadatas": [dict(value or {}) for value in (vector.get("metadatas") or [])],
        }
        embeddings = list(vector.get("embeddings") or [])
        if len(embeddings) == len(ids):
            kwargs["embeddings"] = embeddings
        vector_store.collection.upsert(**kwargs)

    rows = snapshot.get("lexical") or []
    if rows:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO lexical_documents "
                "(id, document, metadata, index_state, tokens) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            connection.commit()

    # Verify that the restored vector set is exactly the original vector ID set.
    restored_ids, _ = _current_document_records(vector_store, document_id)
    expected_ids = set(ids)
    if set(restored_ids) != expected_ids:
        raise RuntimeError(
            f"Rollback snapshot verification failed for document {document_id}: "
            f"expected={sorted(expected_ids)}, restored={sorted(restored_ids)}"
        )

    previous_state = dict(snapshot.get("state") or {})
    if previous_state:
        previous_state["lease_owner"] = None
        previous_state["lease_expires_at"] = None
        previous_state["heartbeat_at"] = None
        state_store.upsert_document(previous_state)

    try:
        state_store.delete_pages(document_id)
    except Exception:
        pass
    for page in snapshot.get("pages") or []:
        values = dict(page)
        values.pop("document_id", None)
        values.pop("page_number", None)
        state_store.upsert_page(document_id, int(page["page_number"]), **values)


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
