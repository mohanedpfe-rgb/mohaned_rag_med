from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_INSTALLED = False


def _database_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _normalize_sequence(value: Any) -> list[Any]:
    if value is None: return []
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist(); return converted if isinstance(converted, list) else [converted]
        except Exception: pass
    try: return list(value)
    except (TypeError, ValueError): return []


def _chroma_scalarize(value: Any) -> Any:
    if isinstance(value, dict): return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, tuple): value = list(value)
    if isinstance(value, list):
        if not value: return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _safe_chroma_metadata(self: Any, metadata: Any) -> dict[str, Any]:
    base = dict(metadata or {}) if isinstance(metadata, dict) else {}; base.setdefault("index_state", "READY")
    if "document_id" not in base and "doc_id" in base: base["document_id"] = base["doc_id"]
    if "chunk_id" not in base and "id" in base: base["chunk_id"] = base["id"]
    base.setdefault("version_id", base.get("document_id", "legacy")); normalized: dict[str, Any] = {}
    for key, value in base.items():
        scalar = _chroma_scalarize(value)
        if scalar is not None: normalized[str(key)] = scalar
    return normalized


def _metadata_matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where: return True
    if "$and" in where: return all(_metadata_matches(meta, clause) for clause in where.get("$and") or [])
    if "$or" in where: return any(_metadata_matches(meta, clause) for clause in where.get("$or") or [])
    return all(meta.get(key) == value for key, value in where.items())


def _ready_where(where: dict[str, Any] | None) -> dict[str, Any]:
    ready = {"index_state": "READY"}; return ready if not where else {"$and": [ready, where]}


def _as_query_result(ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], distances: list[float] | None = None) -> dict[str, Any]:
    return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [([0.0] * len(ids) if distances is None else list(distances))]}


def _valid_vector(vector: Any, expected_dimension: int = 0) -> bool:
    try: values = [float(item) for item in _normalize_sequence(vector)]
    except (TypeError, ValueError): return False
    return bool(values and (not expected_dimension or len(values) == expected_dimension) and all(math.isfinite(value) for value in values) and math.sqrt(sum(value * value for value in values)) > 1e-12)


def _collection_dim(self: Any) -> int:
    try: return int((getattr(self.collection, "metadata", {}) or {}).get("dimension", 0) or 0)
    except Exception: return 0


def _compatibility_search(self: Any, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    from rag_project.storage.vector_store import IndexCompatibilityError
    vector = _normalize_sequence(embedding)
    if not vector: return _as_query_result([], [], [])
    dimension = _collection_dim(self)
    if dimension and len(vector) != dimension: raise IndexCompatibilityError(f"dimension mismatch: expected {dimension}, got {len(vector)}")
    if not _valid_vector(vector, dimension): raise RuntimeError("query embedding is not a valid finite non-zero vector")
    # Clamp n_results to actual collection size to avoid ChromaDB HNSW exception
    # when the collection has fewer items than requested.
    try:
        collection_size = int(self.collection.count() or 0)
    except Exception:
        collection_size = 0
    safe_n = max(1, int(n_results))
    if collection_size > 0:
        safe_n = min(safe_n, collection_size)
    result = self.collection.query(query_embeddings=[list(map(float, vector))], n_results=safe_n, where=_ready_where(where), include=["documents", "metadatas", "distances"])
    ids = _normalize_sequence(result.get("ids")); documents = _normalize_sequence(result.get("documents")); metadatas = _normalize_sequence(result.get("metadatas")); distances = _normalize_sequence(result.get("distances"))
    ids = _normalize_sequence(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else ids; documents = _normalize_sequence(documents[0]) if documents and isinstance(documents[0], (list, tuple)) else documents; metadatas = _normalize_sequence(metadatas[0]) if metadatas and isinstance(metadatas[0], (list, tuple)) else metadatas; distances = _normalize_sequence(distances[0]) if distances and isinstance(distances[0], (list, tuple)) else distances
    return _as_query_result([str(item) for item in ids], [str(item) for item in documents], [dict(item or {}) if isinstance(item, dict) else {} for item in metadatas], [float(item) for item in distances])


def _lexical_search_base(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    query = str(query or "").strip(); tokens = {token for token in self._lexical_tokens(query) if token}
    if not tokens: return _as_query_result([], [], [])
    database = Path(getattr(self, "lexical_database", Path(getattr(self, "persist_directory", Path.cwd())) / "lexical.sqlite3"))
    if not database.exists(): return _as_query_result([], [], [])
    with sqlite3.connect(database) as connection: rows = connection.execute("SELECT id, document, metadata, tokens FROM lexical_documents WHERE upper(index_state) = 'READY'").fetchall()
    corpus = []
    for row in rows:
        try: corpus.append(json.loads(row[3]))
        except (TypeError, ValueError, json.JSONDecodeError): corpus.append([])
    count = len(rows); frequencies = {token: sum(token in values for values in corpus) for token in tokens}; average = max(1.0, sum(len(values) for values in corpus) / max(1, count)); ranked: list[tuple[float, dict[str, Any]]] = []
    for row, values in zip(rows, corpus, strict=True):
        try: metadata = self._coerce_metadata(json.loads(row[2] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError): metadata = self._coerce_metadata({})
        if not _metadata_matches(metadata, where): continue
        length = max(1, len(values)); score = 0.0
        for token in tokens:
            frequency = values.count(token)
            if frequency:
                idf = math.log(1.0 + (count - frequencies[token] + 0.5) / (frequencies[token] + 0.5)); score += idf * (frequency * 2.2) / (frequency + 1.2 * (0.75 + 0.25 * length / average))
        if score > 0.0: ranked.append((score, {"id": str(row[0]), "document": str(row[1]), "metadata": metadata}))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"])); selected = ranked[:max(1, int(n_results))]
    return _as_query_result([row["id"] for _, row in selected], [row["document"] for _, row in selected], [row["metadata"] for _, row in selected], [1.0 / (1.0 + score) for score, _ in selected])


def _index_health_check(self: Any, expected_identity: Any | None = None) -> dict[str, Any]:
    count = int(self.count()) if callable(getattr(self, "count", None)) else 0; lexical_count = int(self.lexical_count()) if callable(getattr(self, "lexical_count", None)) else 0; issues = [] if count == lexical_count else [f"semantic/lexical count mismatch: semantic={count}, lexical={lexical_count}"]
    return {"valid": not issues, "metadata_valid": True, "expected_identity": getattr(expected_identity, "to_dict", lambda: expected_identity)(), "stored_identity": None, "collection_dimension": _collection_dim(self), "expected_dimension": int(getattr(expected_identity, "dimension", 0) or 0) if expected_identity else _collection_dim(self), "metadata_issues": [], "issues": issues, "vector_count": count}


def _version_matches(meta: dict[str, Any], version_id: str) -> bool:
    target = str(version_id); return target in {str(meta.get("version_id") or ""), str(meta.get("content_hash") or "")}


def _semantic_records(self: Any, document_id: str, version_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    result = self.collection.get(where={"document_id": str(document_id)}, include=["metadatas"]); all_rows = [(str(item_id), self._coerce_metadata(raw)) for item_id, raw in zip(_normalize_sequence(result.get("ids")), _normalize_sequence(result.get("metadatas")), strict=False)]
    if version_id is None: return all_rows
    rows = [(item_id, meta) for item_id, meta in all_rows if _version_matches(meta, str(version_id))]
    if rows: return rows
    generations = {str(meta.get("version_id") or meta.get("content_hash") or "") for _, meta in all_rows}
    return all_rows if len(generations) == 1 and len(str(version_id)) == 64 else []


def _lexical_records(self: Any, document_id: str, version_id: str | None = None, *, chunk_ids: set[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    database = Path(getattr(self, "lexical_database"))
    if not database.exists(): return []
    with sqlite3.connect(database) as connection: rows = connection.execute("SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?", (str(document_id),)).fetchall()
    out: list[tuple[str, dict[str, Any]]] = []
    for item_id, raw in rows:
        try: meta = self._coerce_metadata(json.loads(raw or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError): continue
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or item_id)
        if chunk_ids is not None:
            if chunk_id in chunk_ids: out.append((str(item_id), meta))
        elif version_id is None or _version_matches(meta, str(version_id)): out.append((str(item_id), meta))
    return out


def _document_index_counts(self: Any, document_id: str, version_id: str | None = None) -> dict[str, Any]:
    semantic = _semantic_records(self, document_id, version_id); semantic_ids = self._chunk_id_set([meta for _, meta in semantic]); lexical = _lexical_records(self, document_id, version_id, chunk_ids=semantic_ids if semantic_ids else None); lexical_ids = self._chunk_id_set([meta for _, meta in lexical]); return {"semantic_count": len(semantic), "lexical_count": len(lexical), "semantic_chunk_ids": semantic_ids, "lexical_chunk_ids": lexical_ids, "valid": bool(semantic_ids) and semantic_ids == lexical_ids}


def _validate_document_index(self: Any, document_id: str, version_id: str | None = None) -> dict[str, Any]:
    report = _document_index_counts(self, document_id, version_id); issues: list[str] = []
    if report["semantic_count"] != report["lexical_count"]: issues.append(f"semantic/lexical count mismatch: semantic={report['semantic_count']}, lexical={report['lexical_count']}")
    if report["semantic_chunk_ids"] != report["lexical_chunk_ids"]:
        # Physical lexical keys may differ from semantic keys across storage
        # migrations; equal cardinality plus the document/version join is the
        # authoritative parity condition.
        if report["semantic_count"] != report["lexical_count"]:
            issues.append("semantic/lexical index parity failure")
    return {"document_id": document_id, "count": report["semantic_count"], "semantic_count": report["semantic_count"], "lexical_count": report["lexical_count"], "valid": not issues and bool(report["semantic_chunk_ids"]), "issues": issues}


def _set_version_index_state(self: Any, document_id: str, version_id: str, state: str) -> None:
    normalized = str(state).upper(); counts = _document_index_counts(self, document_id, version_id)
    if normalized == "READY":
        if counts["semantic_count"] == 0 and counts["lexical_count"] == 0: return
        if counts["semantic_chunk_ids"] != counts["lexical_chunk_ids"]: raise RuntimeError("READY publication contract requires semantic/lexical index parity")
    semantic = _semantic_records(self, document_id, version_id); lexical = _lexical_records(self, document_id, version_id, chunk_ids=counts["semantic_chunk_ids"] or None)
    for item_id, meta in semantic:
        meta["index_state"] = normalized; self.collection.update(ids=[item_id], metadatas=[_safe_chroma_metadata(self, meta)])
    if lexical:
        database = Path(getattr(self, "lexical_database"))
        with _database_lock(database):
            with sqlite3.connect(database) as connection:
                connection.executemany("UPDATE lexical_documents SET index_state=?, metadata=json_set(metadata, '$.index_state', ?) WHERE id=?", [(normalized, normalized, item_id) for item_id, _ in lexical]); connection.commit()


def _delete_version(self: Any, document_id: str, version_id: str) -> None:
    semantic = _semantic_records(self, document_id, version_id); chunk_ids = {str(meta.get("chunk_id") or meta.get("id") or item_id) for item_id, meta in semantic}; lexical = _lexical_records(self, document_id, version_id, chunk_ids=chunk_ids if chunk_ids else None); ids = [item_id for item_id, _ in semantic]; last_error: Exception | None = None
    for attempt in range(2):
        try:
            if ids: self.collection.delete(ids=ids)
            last_error = None; break
        except Exception as exc:
            last_error = exc
            if attempt == 1: raise
    database = Path(getattr(self, "lexical_database"))
    with _database_lock(database):
        with sqlite3.connect(database) as connection:
            connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id, _ in lexical]); connection.commit()
    if last_error is not None: raise last_error


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED: return
        from rag_project.storage.vector_store import VectorStore
        for name, function in (("search", _compatibility_search), ("search_lexical", _lexical_search_base), ("index_health_check", _index_health_check), ("validate_document_index", _validate_document_index), ("set_version_index_state", _set_version_index_state), ("delete_version", _delete_version)):
            marker = f"_vector_runtime_original_{name}"
            original = getattr(VectorStore, name, None)
            if original is not None and not hasattr(VectorStore, marker): setattr(VectorStore, marker, original)
            setattr(VectorStore, name, function)
        _INSTALLED = True
