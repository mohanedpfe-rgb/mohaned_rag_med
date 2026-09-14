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
    """Convert metadata to values accepted by Chroma without empty-list failures."""
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


def _metadata_matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_metadata_matches(meta, clause) for clause in where.get("$and") or [])
    if "$or" in where:
        return any(_metadata_matches(meta, clause) for clause in where.get("$or") or [])
    return all(meta.get(key) == value for key, value in where.items())


def _as_query_result(ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], distances: list[float] | None = None) -> dict[str, Any]:
    if distances is None:
        distances = [0.0] * len(ids)
    return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [distances]}


def _compatibility_index_health(self: Any, expected_identity: Any | None) -> dict[str, Any]:
    report = self.compatibility_report(expected_identity)
    collection_count = self.count()
    metadata_issues = list(report.get("issues", []))
    issues = [*metadata_issues]
    if collection_count == 0:
        issues.append("Collection is empty.")
    try:
        records = self.collection.get(include=["embeddings"])
        embeddings = _normalize_sequence(records.get("embeddings"))
        invalid = sum(
            1
            for vector in embeddings
            if not self._valid_vector(vector, report.get("collection_dimension") or 0)
        )
        if invalid:
            issues.append(f"{invalid} invalid semantic embeddings.")
    except Exception as exc:
        issues.append(f"Unable to validate stored embeddings: {type(exc).__name__}: {exc}")
    return {
        "valid": not issues,
        "metadata_valid": report.get("metadata_valid", False),
        "expected_identity": getattr(expected_identity, "to_dict", lambda: expected_identity)(),
        "stored_identity": report.get("stored_identity"),
        "collection_dimension": report.get("collection_dimension"),
        "expected_dimension": report.get("expected_dimension"),
        "metadata_issues": metadata_issues,
        "issues": issues,
        "vector_count": collection_count,
    }


def _compatibility_search(self: Any, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = self.expected_identity
    report = self.compatibility_report(expected)
    if not report["valid"]:
        raise type(self).IndexCompatibilityError(report["message"]) if hasattr(type(self), "IndexCompatibilityError") else RuntimeError(report["message"])
    embedding_list = _normalize_sequence(embedding)
    if not embedding_list:
        return _as_query_result([], [], [])
    collection_dim = self._collection_dim()
    if collection_dim and len(embedding_list) != collection_dim:
        raise RuntimeError(f"dimension mismatch: expected {collection_dim}, got {len(embedding_list)}")
    if not self._valid_vector(embedding_list, collection_dim or 0):
        raise RuntimeError("query embedding is not a valid finite non-zero vector")
    results = self.collection.query(
        query_embeddings=[list(map(float, embedding_list))],
        n_results=max(1, int(n_results)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    raw_ids = _normalize_sequence(results.get("ids"))
    raw_documents = _normalize_sequence(results.get("documents"))
    raw_metadatas = _normalize_sequence(results.get("metadatas"))
    raw_distances = _normalize_sequence(results.get("distances"))
    ids = _normalize_sequence(raw_ids[0]) if raw_ids and isinstance(raw_ids[0], (list, tuple)) else raw_ids
    documents = _normalize_sequence(raw_documents[0]) if raw_documents and isinstance(raw_documents[0], (list, tuple)) else raw_documents
    metadatas = _normalize_sequence(raw_metadatas[0]) if raw_metadatas and isinstance(raw_metadatas[0], (list, tuple)) else raw_metadatas
    distances = _normalize_sequence(raw_distances[0]) if raw_distances and isinstance(raw_distances[0], (list, tuple)) else raw_distances
    return _as_query_result(
        [str(item) for item in ids],
        [str(item) for item in documents],
        [dict(item or {}) if isinstance(item, dict) else {} for item in metadatas],
        [float(item) for item in distances],
    )


def _lexical_search_base(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return _as_query_result([], [], [])
    tokens = {token for token in self._lexical_tokens(query) if token}
    if not tokens:
        return _as_query_result([], [], [])
    with sqlite3.connect(self.lexical_database) as connection:
        records = connection.execute(
            "SELECT id, document, metadata, tokens FROM lexical_documents WHERE index_state = 'READY'"
        ).fetchall()
    corpus = [json.loads(row[3]) for row in records]
    document_count = len(records)
    document_frequency = {token: sum(token in row_tokens for row_tokens in corpus) for token in tokens}
    average_length = max(1.0, sum(len(item) for item in corpus) / max(1, document_count))
    rank: list[tuple[float, dict[str, Any]]] = []
    for row, row_tokens in zip(records, corpus, strict=True):
        meta = self._coerce_metadata(json.loads(row[2]))
        if str(meta.get("index_state", "READY")).upper() != "READY":
            continue
        if not _metadata_matches(meta, where):
            continue
        term_counts = {token: row_tokens.count(token) for token in tokens}
        length = max(1, len(row_tokens))
        score = 0.0
        for token, frequency in term_counts.items():
            if not frequency:
                continue
            idf = math.log(1.0 + (document_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            score += idf * (frequency * 2.2) / (frequency + 1.2 * (0.75 + 0.25 * length / average_length))
        if score > 0.0:
            rank.append((score, {"id": str(row[0]), "document": str(row[1]), "metadata": meta}))
    if not rank:
        return _as_query_result([], [], [])
    ranked = sorted(rank, key=lambda item: item[0], reverse=True)[: max(1, int(n_results))]
    return _as_query_result(
        [entry["id"] for _, entry in ranked],
        [entry["document"] for _, entry in ranked],
        [entry["metadata"] for _, entry in ranked],
        [1.0 / (1.0 + score) for score, _ in ranked],
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore
    from rag_project.storage.vector_store import IndexCompatibilityError

    # The storage implementation and runtime hardening must agree on one public
    # API. Older master revisions accidentally deleted this suffix from
    # VectorStore, so restore the methods before installing decorators.
    if not hasattr(VectorStore, "index_health_check"):
        VectorStore.index_health_check = _compatibility_index_health
    if not hasattr(VectorStore, "_as_query_result"):
        VectorStore._as_query_result = staticmethod(_as_query_result)
    if not hasattr(VectorStore, "_metadata_matches"):
        VectorStore._metadata_matches = staticmethod(_metadata_matches)
    if not hasattr(VectorStore, "search"):
        VectorStore.search = _compatibility_search
    if not hasattr(VectorStore, "search_lexical"):
        VectorStore.search_lexical = _lexical_search_base
    if not hasattr(VectorStore, "set_version_state"):
        VectorStore.set_version_state = lambda self, document_id, version_id, state: self.set_version_index_state(document_id, version_id, state)
    if not hasattr(VectorStore, "verify_index"):
        VectorStore.verify_index = lambda self, document_id=None: self.validate_document_index(document_id) if document_id else self.index_health_check(self.expected_identity)
    if not hasattr(VectorStore, "rebuild_index"):
        def rebuild_index(self, document_id=None):
            reconciled = self.reconcile_index(document_id)
            health = self.index_health_check(self.expected_identity)
            return {"status": "RECONCILED" if health.get("valid") else "NEEDS_ATTENTION", "reconciled": reconciled, "health": health}
        VectorStore.rebuild_index = rebuild_index

    original_init = VectorStore.__init__
    original_resolve_dimension = VectorStore._resolve_dimension
    original_coerce_metadata = VectorStore._coerce_metadata
    original_search_lexical = VectorStore.search_lexical

    def hardened_init(self: Any, persist_directory: str | Path, collection_name: str = "rag_documents") -> None:
        original_init(self, persist_directory, collection_name)
        self._runtime_closed = False
        with _database_lock(Path(self.lexical_database)):
            with _connect(Path(self.lexical_database)):
                pass

    def hardened_close(self: Any) -> None:
        if getattr(self, "_runtime_closed", False):
            return
        self._runtime_closed = True
        self.collection = None
        self.client = None

    def hardened_enter(self: Any) -> Any:
        if getattr(self, "_runtime_closed", False):
            raise RuntimeError("VectorStore cannot be re-entered after close().")
        return self

    def hardened_exit(self: Any, exc_type: Any, exc: Any, traceback: Any) -> None:
        hardened_close(self)

    def hardened_resolve_dimension(self: Any, embeddings: Any = None) -> int:
        if embeddings is None or len(_normalize_sequence(embeddings)) == 0:
            stored = int(self._collection_dim() or 0)
            if stored <= 0:
                return 0
        return original_resolve_dimension(self, embeddings)

    def hardened_search_lexical(self: Any, query: str, n_results: int = 5, where=None):
        if getattr(self, "_runtime_closed", False):
            raise RuntimeError("VectorStore is closed.")
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
                if keep:
                    return result
                return result
            return result

    VectorStore._original_coerce_metadata = original_coerce_metadata
    VectorStore._coerce_metadata = _safe_chroma_metadata
    VectorStore.__init__ = hardened_init
    VectorStore.close = hardened_close
    VectorStore.__enter__ = hardened_enter
    VectorStore.__exit__ = hardened_exit
    VectorStore._resolve_dimension = hardened_resolve_dimension
    VectorStore.search_lexical = hardened_search_lexical

    for method_name in ("_upsert_lexical_records", "add_lexical_documents", "set_document_index_state", "set_version_index_state", "delete_version", "clear_all"):
        original = getattr(VectorStore, method_name)
        def make_wrapper(function: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                if getattr(self, "_runtime_closed", False):
                    raise RuntimeError("VectorStore is closed.")
                with _database_lock(Path(self.lexical_database)):
                    return function(self, *args, **kwargs)
            return wrapped
        setattr(VectorStore, method_name, make_wrapper(original))
    _INSTALLED = True


install()

__all__ = ["install"]
