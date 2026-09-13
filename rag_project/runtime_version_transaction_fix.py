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
    """Read all Chroma records and filter by document identity in Python."""
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


def _reopen_vector_store(vector_store: Any) -> None:
    """Reopen the persistent Chroma collection after destructive rollback work."""
    persist_directory = Path(getattr(vector_store, "persist_directory")).resolve()
    collection_name = str(getattr(vector_store, "collection_name"))
    expected_identity = getattr(vector_store, "expected_identity", None)
    old_collection = getattr(vector_store, "collection", None)
    old_client = getattr(vector_store, "client", None)
    try:
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
        else:
            client_close = getattr(old_client, "close", None)
            if callable(client_close):
                client_close()
            vector_store.collection = None
            vector_store.client = None
    except Exception:
        pass
    finally:
        del old_collection, old_client
        gc.collect()

    import chromadb

    vector_store.client = chromadb.PersistentClient(path=str(persist_directory))
    vector_store.collection = vector_store.client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store.expected_identity = expected_identity
    gc.collect()


def _purge_document_records(system: Any, document_id: str) -> None:
    """Reduce the live semantic and lexical indexes to zero records for a document."""
    vector_store = getattr(system, "vector_store", None)
    if vector_store is None or not document_id:
        return

    remaining_ids: list[str] = []
    for _attempt in range(3):
        try:
            ids, _metadata = _current_document_records(vector_store, document_id)
            if ids:
                vector_store.collection.delete(ids=ids)
            verify_ids, _ = _current_document_records(vector_store, document_id)
            if not verify_ids:
                remaining_ids = []
                break
            try:
                vector_store.collection.delete(where={"document_id": document_id})
            except Exception:
                pass
            remaining_ids, _ = _current_document_records(vector_store, document_id)
            if not remaining_ids:
                break
        except Exception:
            continue

    gc.collect()
    if remaining_ids:
        raise RuntimeError(
            f"Rollback could not purge live vector records for document {document_id}: {remaining_ids}"
        )

    lexical_database = Path(vector_store.lexical_database)
    with sqlite3.connect(lexical_database) as connection:
        rows = connection.execute(
            "SELECT id, metadata FROM lexical_documents"
        ).fetchall()
        stale_ids: list[str] = []
        for row_id, raw_metadata in rows:
            try:
                metadata = json.loads(raw_metadata or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if str(metadata.get("document_id") or metadata.get("doc_id") or "") == document_id:
                stale_ids.append(str(row_id))
        if stale_ids:
            connection.executemany(
                "DELETE FROM lexical_documents WHERE id = ?",
                [(item_id,) for item_id in stale_ids],
            )
        connection.commit()

        leftovers = connection.execute(
            "SELECT id, metadata FROM lexical_documents"
        ).fetchall()
        unresolved = []
        for row_id, raw_metadata in leftovers:
            try:
                metadata = json.loads(raw_metadata or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if str(metadata.get("document_id") or metadata.get("doc_id") or "") == document_id:
                unresolved.append(str(row_id))
        if unresolved:
            raise RuntimeError(
                f"Rollback could not purge live lexical records for document {document_id}: {unresolved}"
            )

    _reopen_vector_store(vector_store)


def _restore_snapshot(system: Any, snapshot: dict[str, Any]) -> None:
    vector_store = getattr(system, "vector_store", None)
    state_store = getattr(system, "state_store", None)
    if vector_store is None or state_store is None:
        return

    document_id = str(snapshot.get("document_id") or "")
    if not document_id:
        return

    _purge_document_records(system, document_id)

    vector = snapshot.get("vector") or {}
    ids = [str(value) for value in (vector.get("ids") or [])]
    if ids:
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": list(vector.get("documents") or []),
            "metadatas": [dict(value or {}) for value in (vector.get("metadatas") or [])],
        }
        raw_embeddings = vector.get("embeddings")
        embeddings = raw_embeddings.tolist() if hasattr(raw_embeddings, "tolist") else list(raw_embeddings or [])
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

    state_store.delete_pages(document_id)
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
