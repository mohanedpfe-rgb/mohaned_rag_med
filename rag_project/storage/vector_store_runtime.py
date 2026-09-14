from __future__ import annotations

import json
import math
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
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _safe_chroma_metadata(self: Any, metadata: Any) -> dict[str, Any]:
    # Compatibility installers can wrap this method more than once. Calling
    # a saved wrapper here would recurse, so normalize from the input directly.
    base = dict(metadata or {}) if isinstance(metadata, dict) else {}
    base.setdefault("index_state", "READY")
    if "document_id" not in base and "doc_id" in base:
        base["document_id"] = base["doc_id"]
    if "chunk_id" not in base and "id" in base:
        base["chunk_id"] = base["id"]
    base.setdefault("version_id", base.get("document_id", "legacy"))
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
    return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [([0.0] * len(ids) if distances is None else list(distances))]}


def _valid_vector(vector: Any, expected_dimension: int = 0) -> bool:
    try:
        values = [float(item) for item in _normalize_sequence(vector)]
    except (TypeError, ValueError):
        return False
    return bool(values and (not expected_dimension or len(values) == expected_dimension) and all(math.isfinite(item) for item in values) and math.sqrt(sum(item * item for item in values)) > 1e-12)


def _collection_dim(self: Any) -> int:
    try:
        return int((getattr(self.collection, "metadata", {}) or {}).get("dimension", 0) or 0)
    except Exception:
        return 0


def _compatibility_search(self: Any, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    vector = _normalize_sequence(embedding)
    if not vector:
        return _as_query_result([], [], [])
    dimension = _collection_dim(self)
    if dimension and len(vector) != dimension:
        raise RuntimeError(f"dimension mismatch: expected {dimension}, got {len(vector)}")
    if not _valid_vector(vector, dimension):
        raise RuntimeError("query embedding is not a valid finite non-zero vector")
    result = self.collection.query(query_embeddings=[list(map(float, vector))], n_results=max(1, int(n_results)), where=where, include=["documents", "metadatas", "distances"])
    ids = _normalize_sequence(result.get("ids")); documents = _normalize_sequence(result.get("documents")); metadatas = _normalize_sequence(result.get("metadatas")); distances = _normalize_sequence(result.get("distances"))
    ids = _normalize_sequence(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else ids
    documents = _normalize_sequence(documents[0]) if documents and isinstance(documents[0], (list, tuple)) else documents
    metadatas = _normalize_sequence(metadatas[0]) if metadatas and isinstance(metadatas[0], (list, tuple)) else metadatas
    distances = _normalize_sequence(distances[0]) if distances and isinstance(distances[0], (list, tuple)) else distances
    return _as_query_result([str(item) for item in ids], [str(item) for item in documents], [dict(item or {}) if isinstance(item, dict) else {} for item in metadatas], [float(item) for item in distances])


def _lexical_search_base(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    query = str(query or "").strip()
    tokens = {token for token in self._lexical_tokens(query) if token}
    if not tokens:
        return _as_query_result([], [], [])
    database = Path(getattr(self, "lexical_database", Path(getattr(self, "persist_directory", Path.cwd())) / "lexical.sqlite3"))
    if not database.exists():
        return _as_query_result([], [], [])
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT id, document, metadata, tokens FROM lexical_documents WHERE index_state = 'READY'").fetchall()
    corpus = []
    for row in rows:
        try:
            corpus.append(json.loads(row[3]))
        except (TypeError, ValueError, json.JSONDecodeError):
            corpus.append([])
    count = len(rows)
    frequencies = {token: sum(token in values for values in corpus) for token in tokens}
    average = max(1.0, sum(len(values) for values in corpus) / max(1, count))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row, values in zip(rows, corpus, strict=True):
        try:
            metadata = self._coerce_metadata(json.loads(row[2] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = self._coerce_metadata({})
        if str(metadata.get("index_state", "READY")).upper() != "READY" or not _metadata_matches(metadata, where):
            continue
        length = max(1, len(values)); score = 0.0
        for token in tokens:
            frequency = values.count(token)
            if frequency:
                idf = math.log(1.0 + (count - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
                score += idf * (frequency * 2.2) / (frequency + 1.2 * (0.75 + 0.25 * length / average))
        if score > 0.0:
            ranked.append((score, {"id": str(row[0]), "document": str(row[1]), "metadata": metadata}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:max(1, int(n_results))]
    return _as_query_result([row["id"] for _, row in selected], [row["document"] for _, row in selected], [row["metadata"] for _, row in selected], [1.0 / (1.0 + score) for score, _ in selected])


def _index_health_check(self: Any, expected_identity: Any | None = None) -> dict[str, Any]:
    count = int(self.count()) if callable(getattr(self, "count", None)) else 0
    lexical_count = int(self.lexical_count()) if callable(getattr(self, "lexical_count", None)) else 0
    issues = [] if count == lexical_count else [f"semantic/lexical count mismatch: semantic={count}, lexical={lexical_count}"]
    return {"valid": not issues, "metadata_valid": True, "expected_identity": getattr(expected_identity, "to_dict", lambda: expected_identity)(), "stored_identity": None, "collection_dimension": _collection_dim(self), "expected_dimension": int(getattr(expected_identity, "dimension", 0) or 0) if expected_identity else _collection_dim(self), "metadata_issues": [], "issues": issues, "vector_count": count}


def install() -> None:
    global _INSTALLED
    from rag_project.storage.vector_store import VectorStore
    if not hasattr(VectorStore, "_as_query_result"):
        VectorStore._as_query_result = staticmethod(_as_query_result)
    if not hasattr(VectorStore, "_metadata_matches"):
        VectorStore._metadata_matches = staticmethod(_metadata_matches)
    # These are mandatory public storage APIs. Restore them on every install call
    # because another compatibility adapter must never be able to remove them.
    if not hasattr(VectorStore, "search"):
        VectorStore.search = _compatibility_search
    if not hasattr(VectorStore, "search_lexical"):
        VectorStore.search_lexical = _lexical_search_base
    if not hasattr(VectorStore, "index_health_check"):
        VectorStore.index_health_check = _index_health_check

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
