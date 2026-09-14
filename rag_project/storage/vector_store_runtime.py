from __future__ import annotations

import json
import math
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
        base = dict(metadata or {}) if isinstance(metadata, dict) else {}
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
        if scalar is not None:
            normalized[str(key)] = scalar
    return normalized


def _metadata_matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_metadata_matches(meta, clause) for clause in where.get("$and") or [])
    if "$or" in where:
        return any(_metadata_matches(meta, clause) for clause in where.get("$or") or [])
    return all(meta.get(key) == value for key, value in where.items())


def _as_query_result(ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], distances: list[float] | None = None) -> dict[str, Any]:
    distances = [0.0] * len(ids) if distances is None else distances
    return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [distances]}


def install() -> None:
    """Install storage compatibility explicitly; importing this module is side-effect free."""
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore

    if not hasattr(VectorStore, "_as_query_result"):
        VectorStore._as_query_result = staticmethod(_as_query_result)
    if not hasattr(VectorStore, "_metadata_matches"):
        VectorStore._metadata_matches = staticmethod(_metadata_matches)

    original_coerce = VectorStore._coerce_metadata
    if not getattr(original_coerce, "_runtime_storage_metadata_owner", False):
        def coerce_metadata(self: Any, metadata: Any) -> dict[str, Any]:
            return _safe_chroma_metadata(self, metadata)
        coerce_metadata._runtime_storage_metadata_owner = True
        VectorStore._original_coerce_metadata = original_coerce
        VectorStore._coerce_metadata = coerce_metadata

    original_init = VectorStore.__init__
    if not getattr(original_init, "_runtime_storage_lifecycle_owner", False):
        def hardened_init(self: Any, persist_directory: str | Path, collection_name: str = "rag_documents") -> None:
            original_init(self, persist_directory, collection_name)
            self._runtime_closed = False
            self._runtime_storage_lock = _database_lock(Path(self.lexical_database))
        hardened_init._runtime_storage_lifecycle_owner = True
        VectorStore.__init__ = hardened_init

    if not getattr(getattr(VectorStore, "close", None), "_runtime_storage_close_owner", False):
        original_close = getattr(VectorStore, "close", None)
        if callable(original_close):
            def close(self: Any) -> None:
                if getattr(self, "_runtime_closed", False):
                    return
                self._runtime_closed = True
                try:
                    original_close(self)
                finally:
                    self.collection = None
                    self.client = None
            close._runtime_storage_close_owner = True
            VectorStore.close = close
        else:
            def close(self: Any) -> None:
                self._runtime_closed = True
                self.collection = None
                self.client = None
            close._runtime_storage_close_owner = True
            VectorStore.close = close

    if not hasattr(VectorStore, "__enter__"):
        VectorStore.__enter__ = lambda self: self if not getattr(self, "_runtime_closed", False) else (_ for _ in ()).throw(RuntimeError("VectorStore is closed."))
    if not hasattr(VectorStore, "__exit__"):
        VectorStore.__exit__ = lambda self, exc_type, exc, traceback: self.close()

    _INSTALLED = True


__all__ = ["install"]
