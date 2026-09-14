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
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]
        except Exception:
            pass
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _chroma_scalarize(value: Any) -> Any:
    """Convert metadata to values accepted by Chroma without changing non-empty lists."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return None
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _safe_chroma_metadata(self: Any, metadata: Any) -> dict[str, Any]:
    original = getattr(self, "_original_coerce_metadata", None)
    if not callable(original):
        raw = dict(metadata or {}) if isinstance(metadata, dict) else {}
        base = dict(raw)
        base.setdefault("index_state", "READY")
        if "document_id" not in base and "doc_id" in base:
            base["document_id"] = base["doc_id"]
        if "chunk_id" not in base and "id" in base:
            base["chunk_id"] = base["id"]
        base.setdefault("version_id", base.get("document_id", "legacy"))
    else:
        base = original(metadata)
    normalized: dict[str, Any] = {}
    for key, value in dict(base).items():
        scalar = _chroma_scalarize(value)
        if scalar is None:
            continue
        normalized[str(key)] = scalar
    return normalized


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
        "ids": [[str(item[3].get("chunk_id") or item[1]) for item in selected]],
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
            blocked = set(getattr(self, "_nonready_lexical_ids", set()))
            blocked.update(getattr(type(self), "_nonready_lexical_ids_by_db", {}).get(str(self.lexical_database), set()))
            with sqlite3.connect(Path(self.lexical_database)) as connection:
                blocked.update(
                    str(row[0]) for row in connection.execute(
                        "SELECT json_extract(metadata, '$.chunk_id') FROM lexical_documents WHERE upper(index_state) <> 'READY'"
                    ).fetchall() if row[0]
                )
            if blocked and flat_ids:
                keep = [i for i, item_id in enumerate(flat_ids) if str(item_id) not in blocked]
                for key in ("ids", "documents", "metadatas", "distances"):
                    values = _normalize_sequence(result.get(key))
                    first = _normalize_sequence(values[0]) if values and isinstance(values[0], (list, tuple)) else values
                    result[key] = [[first[i] for i in keep]]
                flat_ids = [flat_ids[i] for i in keep]
                if not flat_ids:
                    return result
            if flat_ids:
                return result
            return _lexical_fallback(self, query, n_results=n_results, where=where)

    VectorStore._original_coerce_metadata = original_coerce_metadata
    VectorStore._coerce_metadata = _safe_chroma_metadata
    VectorStore.__init__ = hardened_init
    VectorStore._resolve_dimension = hardened_resolve_dimension
    # VectorStore.validate_document_index remains the single source of truth for
    # semantic/lexical parity, embedding validity, and its public result schema.
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
