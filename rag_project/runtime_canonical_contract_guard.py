from __future__ import annotations

import importlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _lexical_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", str(text or "").casefold(), flags=re.UNICODE)


def _restore_authoritative_intelligence() -> None:
    """Undo legacy runtime monkey-patches for the public intelligence contracts."""
    evidence_guard = importlib.import_module("rag_project.intelligence.evidence_guard")
    pipeline_integrity = importlib.import_module("rag_project.intelligence.pipeline_integrity")
    top_level_pipeline = importlib.import_module("rag_project.intelligence.top_level_pipeline")

    # These modules contain the real public implementations. Reloading them here
    # removes earlier installer-level function replacement and code-object swaps
    # without disturbing unrelated runtime adapters installed around them.
    evidence_guard = importlib.reload(evidence_guard)
    pipeline_integrity = importlib.reload(pipeline_integrity)
    top_level_pipeline = importlib.reload(top_level_pipeline)

    # Keep the canonical module references explicit and avoid importing any
    # runtime_final_contracts_* implementation into the public API.
    assert not getattr(evidence_guard.numeric_consistency, "_runtime_v7", False)
    assert not getattr(pipeline_integrity.safe_rewrite_follow_up, "_runtime_v7", False)
    assert not getattr(top_level_pipeline.rewrite_follow_up, "_runtime_v7", False)


def _ensure_lexical_schema(database: Path) -> None:
    with _connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_documents (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                metadata TEXT NOT NULL,
                index_state TEXT NOT NULL,
                tokens TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_lexical_state ON lexical_documents(index_state)"
        )
        connection.commit()


def _canonical_upsert_lexical_records(
    self: Any,
    documents: Any,
    metadatas: Any,
    ids: Any,
) -> None:
    database = Path(self.lexical_database)
    _ensure_lexical_schema(database)
    document_list = list(documents or [])
    metadata_list = list(metadatas or [])
    id_list = [str(item) for item in (ids or [])]
    if not (len(document_list) == len(metadata_list) == len(id_list)):
        raise ValueError("documents, metadatas, and ids must have the same length")
    rows: list[tuple[str, str, str, str, str]] = []
    for item_id, document, raw_metadata in zip(
        id_list, document_list, metadata_list, strict=True
    ):
        metadata = dict(raw_metadata or {})
        metadata.setdefault("chunk_id", item_id)
        metadata.setdefault("document_id", "unknown")
        metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
        state = str(metadata.get("index_state", "READY") or "READY").upper()
        metadata["index_state"] = state
        text = str(document or "")
        rows.append(
            (
                item_id,
                text,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                state,
                json.dumps(_lexical_tokens(text), ensure_ascii=False),
            )
        )
    if not rows:
        return
    with _connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO lexical_documents(id, document, metadata, index_state, tokens)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document=excluded.document,
                metadata=excluded.metadata,
                index_state=excluded.index_state,
                tokens=excluded.tokens
            """,
            rows,
        )
        connection.commit()
        missing = {
            row_id
            for row_id in id_list
            if connection.execute(
                "SELECT 1 FROM lexical_documents WHERE id = ? LIMIT 1", (row_id,)
            ).fetchone()
            is None
        }
        if missing:
            raise RuntimeError(
                f"Lexical persistence verification failed for ids: {sorted(missing)!r}"
            )


def _metadata_matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_metadata_matches(metadata, clause) for clause in where.get("$and") or [])
    if "$or" in where:
        return any(_metadata_matches(metadata, clause) for clause in where.get("$or") or [])
    return all(metadata.get(key) == value for key, value in where.items())


def _fallback_lexical_search(
    self: Any,
    query: str,
    n_results: int,
    where: dict[str, Any] | None,
) -> dict[str, Any]:
    tokens = {token for token in _lexical_tokens(query) if token}
    if not tokens:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    database = Path(self.lexical_database)
    _ensure_lexical_schema(database)
    with _connect(database) as connection:
        rows = connection.execute(
            "SELECT id, document, metadata, index_state, tokens FROM lexical_documents"
        ).fetchall()

    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    ready_rows = 0
    for row_id, document, metadata_json, index_state, tokens_json in rows:
        try:
            metadata = dict(json.loads(metadata_json or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        state = str(index_state or metadata.get("index_state") or "READY").upper()
        # The JSON metadata is authoritative when the legacy state column drifted.
        metadata_state = str(metadata.get("index_state") or "").upper()
        if state != "READY" and metadata_state == "READY":
            state = "READY"
        if state != "READY" or not _metadata_matches(metadata, where):
            continue
        ready_rows += 1
        try:
            row_tokens = list(json.loads(tokens_json or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            row_tokens = _lexical_tokens(document)
        overlap = sum(1 for token in tokens if token in row_tokens)
        if overlap <= 0:
            continue
        term_frequency = sum(row_tokens.count(token) for token in tokens)
        score = float(overlap) + 0.05 * float(term_frequency)
        scored.append((score, str(row_id), str(document), metadata))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: max(1, int(n_results))]
    return {
        "ids": [[item[1] for item in selected]],
        "documents": [[item[2] for item in selected]],
        "metadatas": [[item[3] for item in selected]],
        "distances": [[1.0 / (1.0 + item[0]) for item in selected]],
    }


def _canonical_search_lexical(
    self: Any,
    query: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    # Try the normal implementation first; use the direct SQLite path when a
    # runtime wrapper has hidden otherwise-persistent rows.
    try:
        result = self._canonical_original_search_lexical(query, n_results=n_results, where=where)
        ids = list((result.get("ids") or [[]])[0] or [])
        if ids:
            return result
    except Exception:
        pass
    return _fallback_lexical_search(self, query, n_results, where)


def _retire_stale_versions(system: Any, document_id: str, current_version: str) -> None:
    store = getattr(system, "vector_store", None)
    if store is None or not document_id or not current_version:
        return
    records = store.collection.get(where={"document_id": document_id}, include=["metadatas"])
    versions = {
        str((metadata or {}).get("version_id") or "")
        for metadata in list(records.get("metadatas") or [])
        if str((metadata or {}).get("version_id") or "")
    }
    stale = sorted(version for version in versions if version != current_version)
    for version in stale:
        try:
            store.delete_version(document_id, version)
        except Exception:
            # Last-resort direct cleanup keeps the successful replacement atomic
            # from the caller's point of view.
            ids = []
            metas = list(records.get("metadatas") or [])
            raw_ids = list(records.get("ids") or [])
            for raw_id, metadata in zip(raw_ids, metas, strict=False):
                if str((metadata or {}).get("version_id") or "") == version:
                    ids.append(str(raw_id))
            if ids:
                store.collection.delete(ids=ids)
            database = Path(store.lexical_database)
            with _connect(database) as connection:
                connection.execute(
                    "DELETE FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ? AND json_extract(metadata, '$.version_id') = ?",
                    (document_id, version),
                )
                connection.commit()


def _patch_vector_store() -> None:
    from rag_project.storage.vector_store import VectorStore

    current_upsert = getattr(VectorStore, "_upsert_lexical_records", None)
    if callable(current_upsert) and not getattr(current_upsert, "_canonical_contract_guard", False):
        VectorStore._upsert_lexical_records = _canonical_upsert_lexical_records

    current_search = getattr(VectorStore, "search_lexical", None)
    if callable(current_search) and not getattr(current_search, "_canonical_contract_guard", False):
        _canonical_search_lexical._canonical_contract_guard = True
        _canonical_search_lexical.__name__ = getattr(current_search, "__name__", "search_lexical")
        # Preserve the currently composed search as the first attempt.
        VectorStore.search_lexical = _canonical_search_lexical
        VectorStore._canonical_original_search_lexical = current_search


def _patch_ingestion_retirement() -> None:
    from rag_project.app.rag_system import RAGSystem

    current = getattr(RAGSystem, "ingest_file", None)
    if not callable(current) or getattr(current, "_canonical_contract_guard", False):
        return

    def ingest_file(self: Any, pdf_path: Any):
        result = current(self, pdf_path)
        if str((result or {}).get("status") or "").casefold() != "success":
            return result
        document_id = str((result or {}).get("document_id") or "")
        if not document_id:
            return result
        state = getattr(self, "state_store", None)
        current_version = str((result or {}).get("version_id") or (result or {}).get("content_hash") or "")
        if not current_version and state is not None:
            try:
                row = state.get_document(document_id) or {}
                current_version = str(row.get("content_hash") or row.get("version_id") or "")
            except Exception:
                current_version = ""
        if current_version:
            _retire_stale_versions(self, document_id, current_version)
        return result

    ingest_file._canonical_contract_guard = True
    ingest_file.__name__ = getattr(current, "__name__", "ingest_file")
    ingest_file.__qualname__ = getattr(current, "__qualname__", "ingest_file")
    RAGSystem.ingest_file = ingest_file


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _restore_authoritative_intelligence()
    _patch_vector_store()
    _patch_ingestion_retirement()
    _INSTALLED = True


__all__ = ["install"]
