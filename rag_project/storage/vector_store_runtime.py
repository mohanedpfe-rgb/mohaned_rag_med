from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_INSTALLED = False


def _database_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _normalize_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _validate_document_index(
    self: Any,
    document_id: str,
    version_id: str | None = None,
) -> dict[str, Any]:
    print(
        f"DEBUG RUNTIME VALIDATE: persist_directory={self.persist_directory}, "
        f"collection_name={self.collection_name}"
    )
    print(
        f"DEBUG RUNTIME VALIDATE INPUT: document_id={document_id}, "
        f"version_id={version_id!r}"
    )
    records = self.collection.get(
        where={"document_id": document_id},
        include=["metadatas", "documents", "embeddings"],
    )
    ids = _normalize_sequence(records.get("ids"))
    metadatas = _normalize_sequence(records.get("metadatas"))
    documents = _normalize_sequence(records.get("documents"))
    embeddings = _normalize_sequence(records.get("embeddings"))
    print(
        f"DEBUG RUNTIME RAW COUNTS: ids={len(ids)}, metadatas={len(metadatas)}, "
        f"documents={len(documents)}, embeddings={len(embeddings)}"
    )
    for index, metadata in enumerate(metadatas):
        if isinstance(metadata, dict):
            record_id = ids[index] if index < len(ids) else None
            print(
                f"DEBUG RUNTIME RECORD: index={index}, "
                f"id={record_id!r}, "
                f"document_id={metadata.get('document_id')!r}, "
                f"version_id={metadata.get('version_id')!r}, "
                f"chunk_id={metadata.get('chunk_id')!r}, "
                f"index_state={metadata.get('index_state')!r}, "
                f"embedding_len={len(embeddings[index]) if index < len(embeddings) and embeddings[index] is not None else None}, "
                f"text_len={len(str(documents[index])) if index < len(documents) else None}"
            )

    if version_id is None:
        selected = list(range(len(ids)))
    else:
        selected = [
            index
            for index, metadata in enumerate(metadatas)
            if isinstance(metadata, dict) and metadata.get("version_id") == version_id
        ]
    print(
        f"DEBUG RUNTIME VERSION MATCH: requested={version_id!r}, selected_indices={selected}"
    )

    issues: list[str] = []
    if not selected:
        result = {
            "document_id": document_id,
            "count": 0,
            "valid": False,
            "issues": ["no matching index records"],
        }
        print(
            f"DEBUG RUNTIME VALIDATE RESULT: count={result['count']}, "
            f"valid={result['valid']}, issues={result['issues']}"
        )
        return result

    selected_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    expected_dimension = int(self._collection_dim() or 0)
    for index in selected:
        if index >= len(metadatas):
            issues.append(f"missing metadata for record index {index}")
            continue
        metadata = self._coerce_metadata(metadatas[index])
        selected_ids.append(str(ids[index]) if index < len(ids) else "")
        chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or "")
        if not chunk_id:
            issues.append("missing chunk_id")
        elif chunk_id in seen_chunk_ids:
            issues.append(f"duplicate chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        if metadata.get("index_state") not in {"READY", "BUILDING"}:
            issues.append(f"unexpected index_state: {metadata.get('index_state')}")
        if index >= len(embeddings):
            issues.append(f"missing semantic embedding for record index {index}")
        elif not self._valid_vector(embeddings[index], expected_dimension):
            issues.append("invalid semantic embedding")
        if index >= len(documents) or not str(documents[index]).strip():
            issues.append(f"missing document text for record index {index}")

    valid = bool(selected_ids) and not issues
    result = {
        "document_id": document_id,
        "count": len(selected_ids),
        "valid": valid,
        "issues": issues,
    }
    print(
        f"DEBUG RUNTIME VALIDATE RESULT: count={result['count']}, "
        f"valid={result['valid']}, issues={result['issues']}"
    )
    return result


def _synchronize_ingestion_version(self: Any, document_id: str) -> None:
    """Make the canonical state-store version_id authoritative in retrieval stores."""
    if not document_id:
        return
    state_document = self.state_store.get_document(document_id)
    if not state_document:
        return
    canonical_version = str(
        state_document.get("version_id") or state_document.get("content_hash") or ""
    )
    if not canonical_version:
        return

    matches = self.collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )
    ids = _normalize_sequence(matches.get("ids"))
    metadatas = _normalize_sequence(matches.get("metadatas"))
    for item_id, metadata in zip(ids, metadatas, strict=False):
        meta = self._coerce_metadata(metadata)
        if str(meta.get("version_id") or "") != canonical_version:
            meta["version_id"] = canonical_version
            self.collection.update(ids=[str(item_id)], metadatas=[meta])

    with _database_lock(Path(self.lexical_database)):
        with _connect(Path(self.lexical_database)) as connection:
            connection.execute(
                "UPDATE lexical_documents "
                "SET metadata = json_set(metadata, '$.version_id', ?) "
                "WHERE json_extract(metadata, '$.document_id') = ?",
                (canonical_version, document_id),
            )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from rag_project.app.rag_system import RAGSystem
    from rag_project.storage.vector_store import VectorStore

    original_init = VectorStore.__init__
    original_resolve_dimension = VectorStore._resolve_dimension
    original_ingest_file = RAGSystem.ingest_file

    def hardened_init(
        self: Any,
        persist_directory: str | Path,
        collection_name: str = "rag_documents",
    ) -> None:
        original_init(self, persist_directory, collection_name)
        with _database_lock(Path(self.lexical_database)):
            with _connect(Path(self.lexical_database)):
                pass

    def hardened_resolve_dimension(self: Any, embeddings: Any = None) -> int:
        if embeddings is None or len(_normalize_sequence(embeddings)) == 0:
            stored = int(self._collection_dim() or 0)
            if stored <= 0:
                return 0
        return original_resolve_dimension(self, embeddings)

    def synchronized_ingest_file(self: Any, pdf_path: Any) -> Any:
        result = original_ingest_file(self, pdf_path)
        if isinstance(result, dict) and result.get("status") == "success":
            _synchronize_ingestion_version(
                self.vector_store,
                str(result.get("document_id") or ""),
            )
        return result

    VectorStore.__init__ = hardened_init
    VectorStore._resolve_dimension = hardened_resolve_dimension
    VectorStore.validate_document_index = _validate_document_index
    RAGSystem.ingest_file = synchronized_ingest_file

    for method_name in (
        "_upsert_lexical_records",
        "add_lexical_documents",
        "set_document_index_state",
        "set_version_index_state",
        "delete_version",
        "clear_all",
    ):
        original = getattr(VectorStore, method_name)

        def make_wrapper(function: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with _database_lock(Path(self.lexical_database)):
                    return function(self, *args, **kwargs)

            return wrapped

        setattr(VectorStore, method_name, make_wrapper(original))

    _INSTALLED = True


install()

__all__ = ["install"]
