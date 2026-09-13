from __future__ import annotations

import json
import re
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


def _chroma_scalarize(value: Any) -> Any:
    """Keep semantic metadata while converting nested structures to Chroma-safe values."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _safe_chroma_metadata(self: Any, metadata: Any) -> dict[str, Any]:
    base = self._original_coerce_metadata(metadata)
    return {str(key): _chroma_scalarize(value) for key, value in dict(base).items()}


def _validate_document_index(
    self: Any,
    document_id: str,
    version_id: str | None = None,
) -> dict[str, Any]:
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas", "documents", "embeddings"])
    ids = _normalize_sequence(records.get("ids")); metadatas = _normalize_sequence(records.get("metadatas")); documents = _normalize_sequence(records.get("documents")); embeddings = _normalize_sequence(records.get("embeddings"))
    if version_id is None:
        selected = list(range(len(ids)))
    else:
        selected = [index for index, metadata in enumerate(metadatas) if isinstance(metadata, dict) and metadata.get("version_id") == version_id]
    issues: list[str] = []
    if not selected:
        return {"document_id": document_id, "count": 0, "valid": False, "issues": ["no matching index records"]}
    selected_ids: list[str] = []; seen_chunk_ids: set[str] = set(); expected_dimension = int(self._collection_dim() or 0)
    for index in selected:
        if index >= len(metadatas): issues.append(f"missing metadata for record index {index}"); continue
        metadata = self._coerce_metadata(metadatas[index]); selected_ids.append(str(ids[index]) if index < len(ids) else "")
        chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or "")
        if not chunk_id: issues.append("missing chunk_id")
        elif chunk_id in seen_chunk_ids: issues.append(f"duplicate chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        if metadata.get("index_state") not in {"READY", "BUILDING"}: issues.append(f"unexpected index_state: {metadata.get('index_state')}")
        if index >= len(embeddings) or not self._valid_vector(embeddings[index], expected_dimension): issues.append("invalid semantic embedding")
        if index >= len(documents) or not str(documents[index]).strip(): issues.append(f"missing document text for record index {index}")
    return {"document_id": document_id, "count": len(selected_ids), "valid": bool(selected_ids) and not issues, "issues": issues}


def _lexical_fallback(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    tokens = [token for token in re.findall(r"\w+", str(query or "").casefold(), flags=re.UNICODE) if token]
    if not tokens:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    token_set = set(tokens)
    rows = []
    with sqlite3.connect(Path(self.lexical_database)) as connection:
        records = connection.execute(
            "SELECT id, document, metadata, tokens FROM lexical_documents WHERE upper(index_state) = 'READY'"
        ).fetchall()
    for item_id, document, metadata_json, tokens_json in records:
        try:
            metadata = self._coerce_metadata(json.loads(metadata_json or "{}"))
            row_tokens = set(json.loads(tokens_json or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if where and not self._metadata_matches(metadata, where):
            continue
        overlap = len(token_set & row_tokens)
        if overlap <= 0:
            continue
        rows.append((overlap, str(item_id), str(document), metadata))
    rows.sort(key=lambda item: (-item[0], item[1]))
    selected = rows[: max(1, int(n_results))]
    return {
        "ids": [[item[1] for item in selected]],
        "documents": [[item[2] for item in selected]],
        "metadatas": [[item[3] for item in selected]],
        "distances": [[1.0 / (1.0 + item[0]) for item in selected]],
    }


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore
    original_init = VectorStore.__init__
    original_resolve_dimension = VectorStore._resolve_dimension
    original_coerce_metadata = VectorStore._coerce_metadata
    original_search_lexical = VectorStore.search_lexical

    def hardened_init(self: Any, persist_directory: str | Path, collection_name: str = "rag_documents") -> None:
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

    def hardened_search_lexical(self: Any, query: str, n_results: int = 5, where=None):
        with _database_lock(Path(self.lexical_database)):
            result = original_search_lexical(self, query, n_results=n_results, where=where)
            ids = _normalize_sequence(result.get("ids"))
            flat_ids = _normalize_sequence(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else ids
            if flat_ids:
                return result
            return _lexical_fallback(self, query, n_results=n_results, where=where)

    VectorStore._original_coerce_metadata = original_coerce_metadata
    VectorStore._coerce_metadata = _safe_chroma_metadata
    VectorStore.__init__ = hardened_init
    VectorStore._resolve_dimension = hardened_resolve_dimension
    VectorStore.validate_document_index = _validate_document_index
    VectorStore.search_lexical = hardened_search_lexical

    for method_name in ("_upsert_lexical_records", "add_lexical_documents", "set_document_index_state", "set_version_index_state", "delete_version", "clear_all"):
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
