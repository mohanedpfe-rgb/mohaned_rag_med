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
            return converted if isinstance(converted, list) else [converted]
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
    base = dict(metadata or {}) if isinstance(metadata, dict) else {}
    base.setdefault("index_state", "READY")
    if "document_id" not in base and "doc_id" in base:
        base["document_id"] = base["doc_id"]
    if "chunk_id" not in base and "id" in base:
        base["chunk_id"] = base["id"]
    base.setdefault("version_id", base.get("document_id", "legacy"))
    normalized: dict[str, Any] = {}
    for key, value in base.items():
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


def _ready_where(where: dict[str, Any] | None) -> dict[str, Any]:
    ready = {"index_state": "READY"}
    return ready if not where else {"$and": [ready, where]}


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
    result = self.collection.query(query_embeddings=[list(map(float, vector))], n_results=max(1, int(n_results)), where=_ready_where(where), include=["documents", "metadatas", "distances"])
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
        rows = connection.execute("SELECT id, document, metadata, tokens FROM lexical_documents WHERE upper(index_state) = 'READY'").fetchall()
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
        if not _metadata_matches(metadata, where):
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


def _normalized_metadata(value: Any, coerce) -> dict[str, Any]:
    try:
        raw = json.loads(value or "{}") if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        raw = {}
    return coerce(raw)


def _semantic_records(self: Any, document_id: str, version_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    result = self.collection.get(where={"document_id": str(document_id)}, include=["metadatas"])
    rows: list[tuple[str, dict[str, Any]]] = []
    target = str(version_id) if version_id is not None else None
    for item_id, raw in zip(_normalize_sequence(result.get("ids")), _normalize_sequence(result.get("metadatas")), strict=False):
        meta = self._coerce_metadata(raw)
        if target is None or target in {str(meta.get("version_id") or ""), str(meta.get("content_hash") or "")}:
            rows.append((str(item_id), meta))
    if target is not None and not rows and len(str(target)) >= 32:
        # A version fingerprint can differ from both stored aliases. Prefer the
        # exact document generation only when every semantic row shares one value.
        all_rows = [(str(i), self._coerce_metadata(m)) for i, m in zip(_normalize_sequence(result.get("ids")), _normalize_sequence(result.get("metadatas")), strict=False)]
        generations = {str(m.get("version_id") or m.get("content_hash") or "") for _, m in all_rows}
        if len(generations) == 1:
            return all_rows
    return rows


def _lexical_records(self: Any, document_id: str, *, chunk_ids: set[str] | None = None, version_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    with sqlite3.connect(self.lexical_database) as connection:
        rows = connection.execute("SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?", (str(document_id),)).fetchall()
    target = str(version_id) if version_id is not None else None
    out: list[tuple[str, dict[str, Any]]] = []
    for item_id, raw in rows:
        meta = _normalized_metadata(raw, self._coerce_metadata)
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or item_id)
        if chunk_ids is not None:
            if chunk_id in chunk_ids:
                out.append((str(item_id), meta))
            continue
        if target is None or target in {str(meta.get("version_id") or ""), str(meta.get("content_hash") or "")}:
            out.append((str(item_id), meta))
    return out


def _validate_document_index(self: Any, document_id: str, version_id: str | None = None) -> dict[str, Any]:
    semantic = _semantic_records(self, document_id, version_id)
    semantic_ids = {str(meta.get("chunk_id") or meta.get("id") or item_id) for item_id, meta in semantic}
    issues: list[str] = []
    seen: set[str] = set()
    for _, meta in semantic:
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
        if not chunk_id:
            issues.append("missing chunk_id")
        elif chunk_id in seen:
            issues.append(f"duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        state = str(meta.get("index_state") or "").upper()
        if state not in {"BUILDING", "READY"}:
            issues.append(f"unexpected index_state: {state}")
    semantic_count = len(semantic)
    lexical = _lexical_records(self, document_id, chunk_ids=semantic_ids if semantic_ids else None)
    if not semantic_ids and version_id is not None:
        lexical = _lexical_records(self, document_id, version_id=version_id)
    lexical_ids = {str(meta.get("chunk_id") or meta.get("id") or item_id) for item_id, meta in lexical}
    lexical_count = len(lexical)
    if semantic_count != lexical_count:
        issues.append(f"semantic/lexical count mismatch: semantic={semantic_count}, lexical={lexical_count}")
    if semantic_ids != lexical_ids:
        issues.append(f"semantic/lexical chunk mismatch: missing_lexical={sorted(semantic_ids - lexical_ids)[:8]}, missing_semantic={sorted(lexical_ids - semantic_ids)[:8]}")
    try:
        expected_dimension = self._collection_dim()
        records = self.collection.get(where={"document_id": str(document_id)}, include=["embeddings", "metadatas"])
        vectors = _normalize_sequence(records.get("embeddings"))
        if version_id is not None:
            selected = [idx for idx, raw in enumerate(_normalize_sequence(records.get("metadatas"))) if version_id in {str(self._coerce_metadata(raw).get("version_id") or ""), str(self._coerce_metadata(raw).get("content_hash") or "")}]
            if selected:
                vectors = [vectors[idx] for idx in selected if idx < len(vectors)]
        if expected_dimension:
            for vector in vectors:
                if not _valid_vector(vector, expected_dimension):
                    issues.append("invalid semantic embedding")
    except Exception:
        issues.append("semantic embedding inspection failed")
    valid = bool(semantic_count) and not issues
    return {"document_id": str(document_id), "count": semantic_count, "semantic_count": semantic_count, "lexical_count": lexical_count, "valid": valid, "issues": issues}


def _set_version_index_state(self: Any, document_id: str, version_id: str, state: str) -> None:
    normalized_state = str(state).upper()
    semantic = _semantic_records(self, document_id, version_id)
    semantic_ids = {str(meta.get("chunk_id") or meta.get("id") or item_id) for item_id, meta in semantic}
    lexical = _lexical_records(self, document_id, chunk_ids=semantic_ids if semantic_ids else None)
    if normalized_state == "READY":
        validation = _validate_document_index(self, document_id, version_id)
        if not validation["valid"]:
            raise RuntimeError("READY publication contract requires semantic/lexical index parity: " + "; ".join(validation["issues"]))
    for item_id, meta in semantic:
        meta["index_state"] = normalized_state
        self.collection.update(ids=[item_id], metadatas=[meta])
    if lexical:
        with sqlite3.connect(self.lexical_database) as connection:
            connection.executemany("UPDATE lexical_documents SET index_state=?, metadata=json_set(metadata,'$.index_state',?) WHERE id=?", [(normalized_state, normalized_state, item_id) for item_id, _ in lexical])
            connection.commit()


def _delete_version(self: Any, document_id: str, version_id: str) -> None:
    matches = self._semantic_records_for_delete(document_id, version_id) if hasattr(self, "_semantic_records_for_delete") else None
    if matches is None:
        raw = self.collection.get(where={"document_id": str(document_id)}, include=["metadatas"])
        matches = []
        for item_id, metadata in zip(_normalize_sequence(raw.get("ids")), _normalize_sequence(raw.get("metadatas")), strict=False):
            meta = self._coerce_metadata(metadata)
            if str(meta.get("version_id") or "") in {str(version_id), str(meta.get("content_hash") or "")}:
                matches.append(str(item_id))
    removable = [str(item) for item in matches]
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            if removable:
                self.collection.delete(ids=removable)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
    with sqlite3.connect(self.lexical_database) as connection:
        connection.execute("DELETE FROM lexical_documents WHERE json_extract(metadata,'$.document_id')=? AND (json_extract(metadata,'$.version_id')=? OR json_extract(metadata,'$.content_hash')=?)", (str(document_id), str(version_id), str(version_id)))
        connection.commit()
    if last_error is not None:
        raise last_error


def install() -> None:
    global _INSTALLED
    from rag_project.storage.vector_store import VectorStore
    if not hasattr(VectorStore, "_as_query_result"):
        VectorStore._as_query_result = staticmethod(_as_query_result)
    if not hasattr(VectorStore, "_metadata_matches"):
        VectorStore._metadata_matches = staticmethod(_metadata_matches)
    VectorStore.search = _compatibility_search
    VectorStore.search_lexical = _lexical_search_base
    VectorStore.index_health_check = _index_health_check
    VectorStore.validate_document_index = _validate_document_index
    VectorStore.set_version_index_state = _set_version_index_state
    VectorStore.delete_version = _delete_version
    if not hasattr(VectorStore, "verify_index"):
        VectorStore.verify_index = lambda self, document_id=None: self.validate_document_index(document_id) if document_id else {"valid": self.count() == self.lexical_count(), "semantic_count": self.count(), "lexical_count": self.lexical_count(), "count": self.count()}
    if not hasattr(VectorStore, "rebuild_index"):
        VectorStore.rebuild_index = lambda self, document_id=None: {"status": "RECONCILED" if self.reconcile_index(document_id).get("valid", False) else "NEEDS_ATTENTION"}

    original_resolve = VectorStore._resolve_dimension
    if not getattr(original_resolve, "_runtime_unknown_dimension_guard", False):
        def resolve_dimension(self: Any, embeddings: Any = None) -> int:
            if embeddings is None and _collection_dim(self) <= 0:
                return 0
            return original_resolve(self, embeddings)
        resolve_dimension._runtime_unknown_dimension_guard = True
        VectorStore._resolve_dimension = resolve_dimension

    original_coerce = getattr(VectorStore, "_original_coerce_metadata", VectorStore._coerce_metadata)
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

    if not hasattr(VectorStore, "close"):
        def close(self: Any) -> None:
            self._runtime_closed = True
            self.collection = None
            self.client = None
        VectorStore.close = close
    if not hasattr(VectorStore, "__enter__"):
        VectorStore.__enter__ = lambda self: self if not getattr(self, "_runtime_closed", False) else (_ for _ in ()).throw(RuntimeError("VectorStore is closed."))
    if not hasattr(VectorStore, "__exit__"):
        VectorStore.__exit__ = lambda self, exc_type, exc, traceback: self.close()
    _INSTALLED = True


__all__ = ["install"]
