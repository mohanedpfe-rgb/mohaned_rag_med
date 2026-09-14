from __future__ import annotations

import json
import os
import re
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


def _flatten_once(value: Any) -> list[Any]:
    values = _as_list(value)
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        return _as_list(values[0])
    return values


def _sqlite_connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _decode_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _canonical_metadata(vector_store: Any, metadata: Any) -> dict[str, Any]:
    source = dict(metadata) if isinstance(metadata, dict) else {}
    decoded = {str(key): _decode_metadata_value(value) for key, value in source.items()}
    coerce = getattr(vector_store, "_coerce_metadata", None)
    if callable(coerce):
        try:
            decoded = dict(coerce(decoded))
        except Exception:
            pass
    decoded.setdefault("index_state", "READY")
    decoded["index_state"] = str(decoded.get("index_state") or "READY").upper()
    return decoded


def _canonical_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_vector_snapshot(system: Any, document_id: str | None) -> dict[str, tuple[str, dict[str, Any]]]:
    vector_store = system.vector_store
    where = {"document_id": str(document_id)} if document_id is not None else None
    records = vector_store.collection.get(
        where=where,
        include=["documents", "metadatas"],
    )
    vector_ids = _flatten_once(records.get("ids"))
    vector_docs = _flatten_once(records.get("documents"))
    vector_meta = _flatten_once(records.get("metadatas"))

    snapshot: dict[str, tuple[str, dict[str, Any]]] = {}
    for index, raw_id in enumerate(vector_ids):
        item_id = str(raw_id)
        raw_metadata = vector_meta[index] if index < len(vector_meta) else {}
        metadata = _canonical_metadata(vector_store, raw_metadata)
        document = str(vector_docs[index]) if index < len(vector_docs) else ""
        if metadata.get("index_state") != "READY":
            continue
        if document_id is not None and str(metadata.get("document_id")) != str(document_id):
            continue
        snapshot[item_id] = (document, metadata)
    return snapshot


def _read_lexical_snapshot(system: Any, document_id: str | None) -> dict[str, tuple[str, dict[str, Any]]]:
    lexical_db = Path(system.vector_store.lexical_database)
    where = ""
    params: tuple[Any, ...] = ()
    if document_id is not None:
        where = " WHERE json_extract(metadata, '$.document_id') = ?"
        params = (str(document_id),)
    with _sqlite_connect(lexical_db) as connection:
        rows = connection.execute(
            "SELECT id, document, metadata, index_state FROM lexical_documents" + where,
            params,
        ).fetchall()

    lexical_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw_id, document, metadata_json, index_state in rows:
        try:
            metadata = json.loads(metadata_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        canonical = _canonical_metadata(system.vector_store, metadata)
        canonical["index_state"] = str(index_state or canonical.get("index_state") or "READY").upper()
        if canonical["index_state"] != "READY":
            continue
        lexical_map[str(raw_id)] = (str(document), canonical)
    return lexical_map


def validate_runtime_contract(system: Any) -> dict[str, Any]:
    vector_store = getattr(system, "vector_store", None)
    missing = [name for name in REQUIRED_VECTOR_METHODS if not callable(getattr(vector_store, name, None))]
    settings = getattr(system, "settings", None)
    required_dirs = {name: getattr(settings, name, None) for name in ("incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir", "log_dir")}
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


def audit_index_consistency(system: Any, document_id: str | None = None) -> dict[str, Any]:
    vector_map = _canonical_vector_snapshot(system, document_id)
    lexical_map = _read_lexical_snapshot(system, document_id)

    vector_ids = set(vector_map)
    lexical_ids = set(lexical_map)
    vector_only = sorted(vector_ids - lexical_ids)
    lexical_only = sorted(lexical_ids - vector_ids)
    mismatched: list[str] = []

    for item_id in sorted(vector_ids & lexical_ids):
        vector_document, vector_metadata = vector_map[item_id]
        lexical_document, lexical_metadata = lexical_map[item_id]
        if vector_document != lexical_document or _canonical_json(vector_metadata) != _canonical_json(lexical_metadata):
            mismatched.append(item_id)

    valid = not vector_only and not lexical_only and not mismatched
    if document_id is not None:
        valid = valid and bool(vector_ids or lexical_ids)

    return {
        "valid": valid,
        "vector_count": len(vector_ids),
        "lexical_count": len(lexical_ids),
        "vector_only": vector_only,
        "lexical_only": lexical_only,
        "metadata_or_content_mismatch": mismatched,
    }


def quick_index_health(system: Any, document_id: str | None = None) -> dict[str, Any]:
    vector_store = system.vector_store
    try:
        vector_count = len(_canonical_vector_snapshot(system, document_id))
        lexical_db = Path(vector_store.lexical_database)
        where = " WHERE upper(index_state) = 'READY'"
        params: tuple[Any, ...] = ()
        if document_id is not None:
            where += " AND json_extract(metadata, '$.document_id') = ?"
            params = (str(document_id),)
        with _sqlite_connect(lexical_db) as connection:
            row = connection.execute("SELECT COUNT(*) FROM lexical_documents" + where, params).fetchone()
        lexical_count = int(row[0] if row else 0)
        return {
            "valid": vector_count == lexical_count,
            "mode": "quick",
            "vector_count": vector_count,
            "lexical_count": lexical_count,
        }
    except Exception as exc:
        return {"valid": False, "mode": "quick", "error": type(exc).__name__}


def repair_index_consistency(system: Any, document_id: str | None = None) -> dict[str, Any]:
    vector_store = system.vector_store
    before = audit_index_consistency(system, document_id)
    expected = _canonical_vector_snapshot(system, document_id)

    if document_id is not None and not expected:
        raise RuntimeError(
            f"No authoritative READY vector records exist for document_id={document_id!r}; "
            "refusing to delete lexical data"
        )

    database = Path(vector_store.lexical_database)
    restored_ids: list[str] = []
    removed_ids: list[str] = []

    with _sqlite_connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if document_id is None:
                existing_rows = connection.execute("SELECT id FROM lexical_documents").fetchall()
            else:
                existing_rows = connection.execute(
                    "SELECT id FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                    (str(document_id),),
                ).fetchall()

            existing_ids = {str(row[0]) for row in existing_rows}
            expected_ids = set(expected)
            removed_ids = sorted(existing_ids - expected_ids)

            if document_id is None:
                connection.execute("DELETE FROM lexical_documents")
            elif removed_ids:
                connection.executemany(
                    "DELETE FROM lexical_documents WHERE id = ?",
                    [(item_id,) for item_id in removed_ids],
                )

            rows = []
            for item_id, (document, metadata) in sorted(expected.items()):
                normalized = dict(metadata)
                normalized["index_state"] = "READY"
                rows.append(
                    (
                        item_id,
                        document,
                        _canonical_json(normalized),
                        "READY",
                        json.dumps(re.findall(r"\w+", document.casefold()), ensure_ascii=False),
                    )
                )
                if item_id not in existing_ids:
                    restored_ids.append(item_id)

            if rows:
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
        except Exception:
            connection.rollback()
            raise

    after = audit_index_consistency(system, document_id)
    if not after["valid"]:
        raise RuntimeError(
            "Lexical parity repair completed but verification failed: "
            + json.dumps(
                {
                    "vector_only": after["vector_only"],
                    "lexical_only": after["lexical_only"],
                    "mismatched": after["metadata_or_content_mismatch"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    return {
        "repaired": not before["valid"],
        "before": before,
        "after": after,
        "removed_ids": removed_ids,
        "restored_ids": restored_ids,
    }


def run_quality_gate(system: Any, *, repair_drift: bool = True, deep_audit: bool | None = None) -> dict[str, Any]:
    contract = validate_runtime_contract(system)
    if not contract["ok"]:
        return {"ready": False, "contract": contract, "index": None}
    if deep_audit is None:
        deep_audit = str(os.getenv("BOOKRAG_DEEP_QUALITY_AUDIT", "false")).strip().lower() in {"1", "true", "yes", "on"}
    quick = quick_index_health(system)
    index = audit_index_consistency(system) if deep_audit else quick
    repair = None
    if repair_drift and not index["valid"]:
        repair = repair_index_consistency(system)
        index = repair["after"] if deep_audit else quick_index_health(system)
    return {"ready": bool(contract["ok"] and index["valid"]), "contract": contract, "index": index, "repair": repair, "audit_mode": "deep" if deep_audit else "quick"}
