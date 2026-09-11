from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


REQUIRED_VECTOR_METHODS = (
    "add_documents",
    "add_lexical_documents",
    "compatibility_report",
    "index_health_check",
    "search",
    "search_lexical",
    "validate_document_index",
)


def _as_list(value: Any) -> list[Any]:
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


def _sqlite_connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def validate_runtime_contract(system: Any) -> dict[str, Any]:
    """Fail fast on broken composition instead of discovering missing APIs mid-ingestion."""
    vector_store = getattr(system, "vector_store", None)
    missing = [
        name for name in REQUIRED_VECTOR_METHODS
        if not callable(getattr(vector_store, name, None))
    ]
    settings = getattr(system, "settings", None)
    required_dirs = {
        name: getattr(settings, name, None)
        for name in (
            "incoming_dir",
            "processed_dir",
            "failed_dir",
            "archive_dir",
            "vector_db_dir",
            "log_dir",
        )
    }
    path_errors: list[str] = []
    for name, value in required_dirs.items():
        try:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                path_errors.append(f"{name} is not a directory")
        except (TypeError, OSError) as exc:
            path_errors.append(f"{name}: {type(exc).__name__}")
    embedding_model = str(getattr(settings, "embedding_model", "") or "").strip()
    ollama_url = str(getattr(settings, "ollama_base_url", "") or "").strip()
    return {
        "ok": not missing and not path_errors and bool(embedding_model) and bool(ollama_url),
        "missing_vector_methods": missing,
        "path_errors": path_errors,
        "embedding_model_configured": bool(embedding_model),
        "ollama_url_configured": bool(ollama_url),
    }


def _collect_consistency_records(system: Any, document_id: str | None) -> dict[str, Any]:
    vector_store = system.vector_store
    lexical_db = Path(vector_store.lexical_database)
    where = {"document_id": document_id} if document_id else None
    records = vector_store.collection.get(
        where=where,
        include=["documents", "metadatas"],
    )
    vector_ids = _as_list(records.get("ids"))
    vector_docs = _as_list(records.get("documents"))
    vector_meta = _as_list(records.get("metadatas"))
    vector_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for index, raw_id in enumerate(vector_ids):
        item_id = str(raw_id)
        metadata = (
            vector_meta[index]
            if index < len(vector_meta) and isinstance(vector_meta[index], dict)
            else {}
        )
        document = vector_docs[index] if index < len(vector_docs) else ""
        vector_map[item_id] = (str(document), dict(metadata))

    with _sqlite_connect(lexical_db) as connection:
        if document_id:
            rows = connection.execute(
                """
                SELECT id, document, metadata, index_state
                FROM lexical_documents
                WHERE json_extract(metadata, '$.document_id') = ?
                """,
                (str(document_id),),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id, document, metadata, index_state FROM lexical_documents"
            ).fetchall()

    lexical_map: dict[str, tuple[str, dict[str, Any], str]] = {}
    for raw_id, document, metadata_json, index_state in rows:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        lexical_map[str(raw_id)] = (
            str(document),
            metadata,
            str(index_state).upper(),
        )
    return {"vector": vector_map, "lexical": lexical_map}


def audit_index_consistency(system: Any, document_id: str | None = None) -> dict[str, Any]:
    """Compare persistent Chroma records against the SQLite lexical mirror."""
    records = _collect_consistency_records(system, document_id)
    vector_map = records["vector"]
    lexical_map = records["lexical"]
    vector_ready = {
        item_id
        for item_id, (_, metadata) in vector_map.items()
        if str(metadata.get("index_state", "READY")).upper() == "READY"
    }
    lexical_ready = {
        item_id
        for item_id, (_, _, state) in lexical_map.items()
        if state == "READY"
    }
    vector_only = sorted(vector_ready - lexical_ready)
    lexical_only = sorted(lexical_ready - vector_ready)
    mismatched: list[str] = []
    for item_id in sorted(vector_ready & lexical_ready):
        vector_doc, vector_metadata = vector_map[item_id]
        lexical_doc, lexical_metadata, _ = lexical_map[item_id]
        if (
            vector_doc != lexical_doc
            or vector_metadata.get("document_id") != lexical_metadata.get("document_id")
            or vector_metadata.get("version_id") != lexical_metadata.get("version_id")
        ):
            mismatched.append(item_id)

    return {
        "valid": not vector_only and not lexical_only and not mismatched,
        "vector_count": len(vector_ready),
        "lexical_count": len(lexical_ready),
        "vector_only": vector_only,
        "lexical_only": lexical_only,
        "metadata_or_content_mismatch": mismatched,
    }


def quick_index_health(system: Any, document_id: str | None = None) -> dict[str, Any]:
    """Perform a bounded startup check without traversing every stored record."""
    vector_store = system.vector_store
    try:
        where = {"document_id": document_id} if document_id else None
        vector_count = int(vector_store.collection.count()) if where is None else len(
            _as_list(vector_store.collection.get(where=where, include=[]).get("ids"))
        )
        lexical_db = Path(vector_store.lexical_database)
        with _sqlite_connect(lexical_db) as connection:
            if document_id:
                row = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM lexical_documents
                    WHERE index_state = 'READY'
                      AND json_extract(metadata, '$.document_id') = ?
                    """,
                    (str(document_id),),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM lexical_documents WHERE index_state = 'READY'"
                ).fetchone()
        lexical_count = int(row[0] if row else 0)
        ready = vector_count == lexical_count
        return {
            "valid": ready,
            "mode": "quick",
            "vector_count": vector_count,
            "lexical_count": lexical_count,
        }
    except Exception as exc:
        return {"valid": False, "mode": "quick", "error": type(exc).__name__}


def repair_index_consistency(system: Any, document_id: str | None = None) -> dict[str, Any]:
    """Rebuild the lexical mirror directly from authoritative vector records and verify every row."""
    vector_store = system.vector_store
    audit = audit_index_consistency(system, document_id)
    scope = str(document_id) if document_id else None
    where = {"document_id": document_id} if document_id else None
    records = vector_store.collection.get(
        where=where,
        include=["documents", "metadatas"],
    )
    vector_ids = _as_list(records.get("ids"))
    vector_docs = _as_list(records.get("documents"))
    vector_meta = _as_list(records.get("metadatas"))

    expected: dict[str, tuple[str, dict[str, Any]]] = {}
    for index, raw_id in enumerate(vector_ids):
        metadata = (
            vector_meta[index]
            if index < len(vector_meta) and isinstance(vector_meta[index], dict)
            else {}
        )
        document = vector_docs[index] if index < len(vector_docs) else ""
        normalized = dict(metadata)
        normalized.setdefault("chunk_id", str(raw_id))
        normalized.setdefault("document_id", scope or "unknown")
        normalized.setdefault("version_id", normalized.get("document_id", "legacy"))
        normalized["index_state"] = str(normalized.get("index_state", "READY") or "READY").upper()
        expected[str(raw_id)] = (str(document), normalized)

    database = Path(vector_store.lexical_database)
    with _sqlite_connect(database) as connection:
        if scope:
            rows = connection.execute(
                "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (scope,),
            ).fetchall()
        else:
            rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
        delete_ids: list[str] = []
        for raw_id, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if scope and str(metadata.get("document_id")) != scope:
                continue
            if str(raw_id) not in expected:
                delete_ids.append(str(raw_id))
        if delete_ids:
            connection.executemany(
                "DELETE FROM lexical_documents WHERE id = ?",
                [(item_id,) for item_id in delete_ids],
            )

        if expected:
            connection.executemany(
                """INSERT INTO lexical_documents(id, document, metadata, index_state, tokens)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document=excluded.document,
                    metadata=excluded.metadata,
                    index_state=excluded.index_state,
                    tokens=excluded.tokens""",
                [
                    (
                        item_id,
                        document,
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        metadata["index_state"],
                        json.dumps(__import__("re").findall(r"\\w+", document.casefold()), ensure_ascii=False),
                    )
                    for item_id, (document, metadata) in expected.items()
                ],
            )
        connection.commit()

        for item_id in expected:
            row = connection.execute(
                "SELECT document, metadata, index_state FROM lexical_documents WHERE id = ?",
                (item_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Lexical repair failed to persist row {item_id!r}")
            persisted_meta = json.loads(row[1] or "{}")
            expected_document, expected_meta = expected[item_id]
            if row[0] != expected_document or persisted_meta.get("version_id") != expected_meta.get("version_id"):
                raise RuntimeError(f"Lexical repair verification failed for row {item_id!r}")

    final = audit_index_consistency(system, document_id)
    return {
        "repaired": not audit["valid"] and final["valid"],
        "before": audit,
        "after": final,
    }


def run_quality_gate(
    system: Any,
    *,
    repair_drift: bool = True,
    deep_audit: bool | None = None,
) -> dict[str, Any]:
    """Run a cheap startup gate and an optional full consistency audit."""
    contract = validate_runtime_contract(system)
    if not contract["ok"]:
        return {"ready": False, "contract": contract, "index": None}

    if deep_audit is None:
        deep_audit = str(os.getenv("BOOKRAG_DEEP_QUALITY_AUDIT", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    quick = quick_index_health(system)
    index = audit_index_consistency(system) if deep_audit else quick
    repair = None
    if repair_drift and not index["valid"]:
        repair = repair_index_consistency(system)
        index = repair["after"] if deep_audit else quick_index_health(system)

    return {
        "ready": bool(contract["ok"] and index["valid"]),
        "contract": contract,
        "index": index,
        "repair": repair,
        "audit_mode": "deep" if deep_audit else "quick",
    }
