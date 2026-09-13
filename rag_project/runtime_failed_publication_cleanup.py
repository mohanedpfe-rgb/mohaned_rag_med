from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

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


def _cleanup_new_publication_records(
    system: Any,
    source: Path,
    content_hash: str | None,
    baseline_vector_ids: set[str],
    baseline_lexical_ids: set[str],
) -> None:
    """Remove only records created after the failed publication began."""
    vector_store = getattr(system, "vector_store", None)
    if vector_store is None:
        return

    try:
        records = vector_store.collection.get(include=["metadatas"])
        removable: list[str] = []
        for item_id, metadata in zip(
            records.get("ids", []) or [],
            records.get("metadatas", []) or [],
            strict=False,
        ):
            record_id = str(item_id)
            if record_id in baseline_vector_ids:
                continue
            meta = dict(metadata or {})
            if _failed_record_matches(record_id, meta, source, content_hash):
                removable.append(record_id)
        if removable:
            vector_store.collection.delete(ids=removable)
    except Exception as exc:
        logger = getattr(system, "logger", None)
        if logger is not None:
            logger.exception("Failed to remove newly-created semantic records after rollback: %s", exc)

    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
            removable: list[str] = []
            for row_id, raw_metadata in rows:
                record_id = str(row_id)
                if record_id in baseline_lexical_ids:
                    continue
                try:
                    metadata = json.loads(raw_metadata or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if _failed_record_matches(record_id, metadata, source, content_hash):
                    removable.append(record_id)
            if removable:
                connection.executemany(
                    "DELETE FROM lexical_documents WHERE id = ?",
                    [(record_id,) for record_id in removable],
                )
            connection.commit()
    except Exception as exc:
        logger = getattr(system, "logger", None)
        if logger is not None:
            logger.exception("Failed to remove newly-created lexical records after rollback: %s", exc)


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
                    self,
                    source,
                    content_hash,
                    baseline_vector_ids,
                    baseline_lexical_ids,
                )
                raise

            status = str(result.get("status") or "").upper() if isinstance(result, dict) else ""
            if status in {"FAILED", "FAILED_INDEXING", "FAILED_EMBEDDING", "FAILED_EXTRACTION"}:
                _cleanup_new_publication_records(
                    self,
                    source,
                    content_hash,
                    baseline_vector_ids,
                    baseline_lexical_ids,
                )
            return result

        ingest_file._failed_publication_cleanup_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", ingest_file.__name__)
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
