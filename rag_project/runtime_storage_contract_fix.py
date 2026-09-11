from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", str(text or "").casefold(), flags=re.UNICODE)


def _ensure_schema(path: Path) -> None:
    with _connect(path) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS lexical_documents (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                metadata TEXT NOT NULL,
                index_state TEXT NOT NULL,
                tokens TEXT NOT NULL
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_lexical_state ON lexical_documents(index_state)")
        db.commit()


def _normalized_rows(documents: Any, metadatas: Any, ids: Any) -> list[tuple[str, str, str, str, str]]:
    documents_list = list(documents or [])
    metadata_list = list(metadatas or [])
    id_list = [str(item) for item in (ids or [])]
    if not (len(documents_list) == len(metadata_list) == len(id_list)):
        raise ValueError("documents, metadatas, and ids must have the same length")
    rows: list[tuple[str, str, str, str, str]] = []
    for item_id, document, raw_meta in zip(id_list, documents_list, metadata_list, strict=True):
        metadata = dict(raw_meta or {})
        metadata.setdefault("chunk_id", item_id)
        metadata.setdefault("document_id", "unknown")
        metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
        metadata["index_state"] = str(metadata.get("index_state", "READY") or "READY").upper()
        text = str(document or "")
        rows.append(
            (
                item_id,
                text,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                metadata["index_state"],
                json.dumps(_tokens(text), ensure_ascii=False),
            )
        )
    return rows


def _upsert(self: Any, documents: Any, metadatas: Any, ids: Any) -> None:
    rows = _normalized_rows(documents, metadatas, ids)
    if not rows:
        return
    database = Path(self.lexical_database)
    _ensure_schema(database)
    with _connect(database) as db:
        db.executemany(
            """INSERT INTO lexical_documents(id, document, metadata, index_state, tokens)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document=excluded.document,
                metadata=excluded.metadata,
                index_state=excluded.index_state,
                tokens=excluded.tokens""",
            rows,
        )
        db.commit()
        missing = [
            row[0]
            for row in rows
            if db.execute("SELECT 1 FROM lexical_documents WHERE id = ?", (row[0],)).fetchone() is None
        ]
        if missing:
            raise RuntimeError(f"Lexical persistence verification failed for ids: {missing!r}")


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, part) for part in where.get("$and") or [])
    if "$or" in where:
        return any(_matches(meta, part) for part in where.get("$or") or [])
    return all(meta.get(key) == value for key, value in where.items())


def _read_ready_rows(self: Any, where: dict[str, Any] | None = None) -> list[tuple[str, str, dict[str, Any]]]:
    database = Path(self.lexical_database)
    _ensure_schema(database)
    with _connect(database) as db:
        rows = db.execute(
            "SELECT id, document, metadata, index_state, tokens FROM lexical_documents"
        ).fetchall()
    ready: list[tuple[str, str, dict[str, Any]]] = []
    for row_id, document, metadata_json, state, _tokens_json in rows:
        try:
            metadata = dict(json.loads(metadata_json or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        effective_state = str(state or metadata.get("index_state") or "READY").upper()
        metadata_state = str(metadata.get("index_state") or "").upper()
        if effective_state != "READY" and metadata_state == "READY":
            effective_state = "READY"
        if effective_state != "READY" or not _matches(metadata, where):
            continue
        ready.append((str(row_id), str(document), metadata))
    return ready


def _search(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    query_tokens = {token for token in _tokens(query) if token}
    if not query_tokens:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    database = Path(self.lexical_database)
    _ensure_schema(database)
    with _connect(database) as db:
        rows = db.execute(
            "SELECT id, document, metadata, index_state, tokens FROM lexical_documents"
        ).fetchall()
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for row_id, document, metadata_json, state, tokens_json in rows:
        try:
            metadata = dict(json.loads(metadata_json or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        effective_state = str(state or metadata.get("index_state") or "READY").upper()
        if effective_state != "READY" and str(metadata.get("index_state") or "").upper() == "READY":
            effective_state = "READY"
        if effective_state != "READY" or not _matches(metadata, where):
            continue
        try:
            row_tokens = list(json.loads(tokens_json or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            row_tokens = _tokens(document)
        overlap = sum(1 for token in query_tokens if token in row_tokens)
        if overlap <= 0:
            continue
        frequency = sum(row_tokens.count(token) for token in query_tokens)
        score = float(overlap) + 0.05 * float(frequency)
        scored.append((score, str(row_id), str(document), metadata))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: max(1, int(n_results))]
    return {
        "ids": [[item[1] for item in selected]],
        "documents": [[item[2] for item in selected]],
        "metadatas": [[item[3] for item in selected]],
        "distances": [[1.0 / (1.0 + item[0]) for item in selected]],
    }


def _get_documents(self: Any, where: dict[str, Any] | None = None) -> dict[str, Any]:
    original = getattr(self, "_storage_contract_original_get_documents", None)
    if callable(original):
        try:
            result = original(self, where)
            documents = result.get("documents") if isinstance(result, dict) else None
            if documents:
                return result
        except Exception:
            pass
    rows = _read_ready_rows(self, where)
    return {
        "ids": [item[0] for item in rows],
        "documents": [item[1] for item in rows],
        "metadatas": [item[2] for item in rows],
    }


def _wrap_write_method(name: str):
    def decorator(original: Any):
        def wrapped(self: Any, *args: Any, **kwargs: Any):
            result = original(self, *args, **kwargs)
            if name == "add_documents":
                if len(args) >= 4:
                    _upsert(self, args[0], args[1], args[3])
                else:
                    _upsert(self, kwargs.get("documents"), kwargs.get("metadatas"), kwargs.get("ids"))
            elif name == "add_lexical_documents":
                if len(args) >= 3:
                    _upsert(self, args[0], args[1], args[2])
                else:
                    _upsert(self, kwargs.get("documents"), kwargs.get("metadatas"), kwargs.get("ids"))
            return result
        wrapped.__name__ = getattr(original, "__name__", name)
        wrapped.__qualname__ = getattr(original, "__qualname__", name)
        wrapped._storage_contract_fix = True
        return wrapped
    return decorator


def _retire(system: Any, document_id: str, current_version: str) -> None:
    store = getattr(system, "vector_store", None)
    if store is None or not document_id or not current_version:
        return
    database = Path(store.lexical_database)
    _ensure_schema(database)
    with _connect(database) as db:
        db.execute(
            "DELETE FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ? AND json_extract(metadata, '$.version_id') <> ?",
            (document_id, current_version),
        )
        db.commit()
    collection = getattr(store, "collection", None)
    if collection is None:
        return
    try:
        records = collection.get(where={"document_id": document_id}, include=["metadatas"])
    except Exception:
        return
    stale_ids = []
    for item_id, metadata in zip(records.get("ids") or [], records.get("metadatas") or [], strict=False):
        if str((metadata or {}).get("version_id") or "") != current_version:
            stale_ids.append(str(item_id))
    if stale_ids:
        collection.delete(ids=stale_ids)


def _prepare_explicit_test_embedding_mode(system: Any) -> None:
    settings = getattr(system, "settings", None)
    service = getattr(system, "embedding_service", None)
    explicit = bool(getattr(settings, "embedding_test_mode", False))
    model_marker = str(getattr(settings, "embedding_model", "") or "").strip().casefold() == "test"
    if not service or not (explicit or model_marker):
        return
    service.test_mode = True
    service.provider = "deterministic-test"
    service.last_error = None
    if getattr(service, "dimension", None) is None:
        service.discover_dimension()
    system.embedding_startup_error = None


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore
    from rag_project.app.rag_system import RAGSystem

    VectorStore._upsert_lexical_records = _upsert
    VectorStore.search_lexical = _search

    current_get_documents = getattr(VectorStore, "get_documents", None)
    if callable(current_get_documents) and not getattr(current_get_documents, "_storage_contract_fix", False):
        VectorStore._storage_contract_original_get_documents = current_get_documents
        VectorStore.get_documents = _get_documents

    for method_name in ("add_documents", "add_lexical_documents"):
        current_method = getattr(VectorStore, method_name, None)
        if callable(current_method) and not getattr(current_method, "_storage_contract_fix", False):
            setattr(VectorStore, method_name, _wrap_write_method(method_name)(current_method))

    current_ingest = getattr(RAGSystem, "ingest_file", None)
    if callable(current_ingest) and not getattr(current_ingest, "_storage_contract_fix", False):
        def ingest_file(self: Any, pdf_path: Any):
            _prepare_explicit_test_embedding_mode(self)
            result = current_ingest(self, pdf_path)
            if str((result or {}).get("status") or "").casefold() == "success":
                document_id = str((result or {}).get("document_id") or "")
                version = str((result or {}).get("content_hash") or (result or {}).get("version_id") or "")
                if not version:
                    try:
                        row = self.state_store.get_document(document_id) or {}
                        version = str(row.get("content_hash") or row.get("version_id") or "")
                    except Exception:
                        version = ""
                if document_id and version:
                    _retire(self, document_id, version)
            return result
        ingest_file._storage_contract_fix = True
        ingest_file.__name__ = getattr(current_ingest, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(current_ingest, "__qualname__", "ingest_file")
        RAGSystem.ingest_file = ingest_file

    _INSTALLED = True


__all__ = ["install"]
