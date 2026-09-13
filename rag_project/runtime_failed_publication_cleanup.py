from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import chromadb

_LOCK = threading.RLock()
_INSTALLED = False


def _source_name_matches(value: Any, source: Path) -> bool:
    name = Path(str(value or "")).name
    return bool(name) and (name == source.name or name.endswith(source.name))


def _file_hash(source: Path) -> str | None:
    try:
        if not source.is_file():
            return None
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _failed_record_matches(
    item_id: str,
    metadata: dict[str, Any],
    source: Path,
    content_hash: str | None,
) -> bool:
    source_marker = metadata.get("file_name") or metadata.get("source_path") or metadata.get("file_path")
    if _source_name_matches(source_marker, source):
        return True
    wanted = str(content_hash or "")
    if not wanted:
        return False
    values = {
        str(metadata.get("version_id") or ""),
        str(metadata.get("content_hash") or ""),
        str(item_id),
        str(metadata.get("chunk_id") or ""),
        str(metadata.get("document_id") or ""),
    }
    return any(wanted in value for value in values)


def _snapshot_vector_ids(vector_store: Any) -> set[str]:
    try:
        records = vector_store.collection.get(include=["metadatas"])
        return {str(item_id) for item_id in records.get("ids", []) or []}
    except Exception:
        return set()


def _snapshot_lexical_ids(vector_store: Any) -> set[str]:
    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            rows = connection.execute("SELECT id FROM lexical_documents").fetchall()
        return {str(row[0]) for row in rows}
    except Exception:
        return set()


def _reopen_vector_store(vector_store: Any) -> None:
    """Reattach the existing VectorStore object to a fresh Chroma client/collection."""
    persist_directory = Path(getattr(vector_store, "persist_directory")).resolve()
    collection_name = str(getattr(vector_store, "collection_name"))
    expected_identity = getattr(vector_store, "expected_identity", None)
    try:
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
        else:
            vector_store.collection = None
            vector_store.client = None
    except Exception:
        try:
            vector_store.collection = None
            vector_store.client = None
        except Exception:
            pass
    gc.collect()
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        clear = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear):
            clear()
    except Exception:
        pass
    vector_store.client = chromadb.PersistentClient(path=str(persist_directory))
    vector_store.collection = vector_store.client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store.expected_identity = expected_identity


def _purge_sqlite_ids(vector_store: Any, ids: set[str]) -> None:
    if not ids:
        return
    with sqlite3.connect(vector_store.lexical_database) as connection:
        connection.executemany(
            "DELETE FROM lexical_documents WHERE id = ?",
            [(item_id,) for item_id in sorted(ids)],
        )
        connection.commit()


def _collect_new_records(
    vector_store: Any,
    source: Path,
    content_hash: str | None,
    baseline_vector_ids: set[str],
) -> set[str]:
    try:
        records = vector_store.collection.get(include=["metadatas"])
    except Exception:
        return set()
    created: set[str] = set()
    for item_id, metadata in zip(
        records.get("ids", []) or [],
        records.get("metadatas", []) or [],
        strict=False,
    ):
        record_id = str(item_id)
        if record_id in baseline_vector_ids:
            continue
        if _failed_record_matches(record_id, dict(metadata or {}), source, content_hash):
            created.add(record_id)
    return created


def _collect_new_lexical_records(
    vector_store: Any,
    source: Path,
    content_hash: str | None,
    baseline_lexical_ids: set[str],
) -> set[str]:
    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
    except Exception:
        return set()
    created: set[str] = set()
    for row_id, raw_metadata in rows:
        record_id = str(row_id)
        if record_id in baseline_lexical_ids:
            continue
        try:
            metadata = json.loads(raw_metadata or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if _failed_record_matches(record_id, metadata, source, content_hash):
            created.add(record_id)
    return created


def _cleanup_new_publication_records(
    system: Any,
    source: Path,
    content_hash: str | None,
    baseline_vector_ids: set[str],
    baseline_lexical_ids: set[str],
) -> None:
    """Roll back a failed publication and prove the failed records are gone."""
    vector_store = getattr(system, "vector_store", None)
    if vector_store is None:
        return

    last_remaining: set[str] = set()
    for _attempt in range(3):
        new_vector_ids = _collect_new_records(
            vector_store, source, content_hash, baseline_vector_ids
        )
        if not new_vector_ids:
            # The Chroma view may already be stale while the lexical store still has
            # physical rows. The lexical cleanup below remains authoritative.
            break
        try:
            vector_store.collection.delete(ids=sorted(new_vector_ids))
        except Exception as exc:
            logger = getattr(system, "logger", None)
            if logger is not None:
                logger.exception("Failed semantic rollback delete: %s", exc)
        gc.collect()
        try:
            _reopen_vector_store(vector_store)
        except Exception as exc:
            logger = getattr(system, "logger", None)
            if logger is not None:
                logger.exception("Failed to reopen Chroma after rollback: %s", exc)
        remaining = _collect_new_records(
            vector_store, source, content_hash, baseline_vector_ids
        )
        last_remaining = remaining
        if not remaining:
            break

    new_lexical_ids = _collect_new_lexical_records(
        vector_store, source, content_hash, baseline_lexical_ids
    )
    if new_lexical_ids:
        try:
            _purge_sqlite_ids(vector_store, new_lexical_ids)
        except Exception as exc:
            logger = getattr(system, "logger", None)
            if logger is not None:
                logger.exception("Failed lexical rollback delete: %s", exc)

    # Re-open after SQL cleanup as well, then perform a final physical invariant check.
    try:
        _reopen_vector_store(vector_store)
    except Exception:
        pass
    final_remaining = _collect_new_records(
        vector_store, source, content_hash, baseline_vector_ids
    )
    if final_remaining or last_remaining:
        raise RuntimeError(
            "Failed publication rollback left physical semantic records: "
            + ", ".join(sorted(final_remaining or last_remaining))
        )

    final_lexical = _collect_new_lexical_records(
        vector_store, source, content_hash, baseline_lexical_ids
    )
    if final_lexical:
        raise RuntimeError(
            "Failed publication rollback left physical lexical records: "
            + ", ".join(sorted(final_lexical))
        )


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.app.rag_system import RAGSystem

        original = getattr(RAGSystem, "ingest_file", None)
        if not callable(original) or getattr(original, "_failed_publication_cleanup_fix", False):
            _INSTALLED = True
            return

        def ingest_file(self: Any, pdf_path: str | Path, *args: Any, **kwargs: Any):
            source = Path(pdf_path)
            vector_store = getattr(self, "vector_store", None)
            content_hash = _file_hash(source)
            baseline_vector_ids = _snapshot_vector_ids(vector_store) if vector_store is not None else set()
            baseline_lexical_ids = _snapshot_lexical_ids(vector_store) if vector_store is not None else set()
            try:
                result = original(self, source, *args, **kwargs)
            except Exception:
                _cleanup_new_publication_records(
                    self, source, content_hash, baseline_vector_ids, baseline_lexical_ids
                )
                raise

            status = str(result.get("status") or "").upper() if isinstance(result, dict) else ""
            if status in {"FAILED", "FAILED_INDEXING", "FAILED_EMBEDDING", "FAILED_EXTRACTION"}:
                _cleanup_new_publication_records(
                    self, source, content_hash, baseline_vector_ids, baseline_lexical_ids
                )
            return result

        ingest_file._failed_publication_cleanup_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", ingest_file.__name__)
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
