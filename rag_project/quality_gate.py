from __future__ import annotations

import json
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


def validate_runtime_contract(system: Any) -> dict[str, Any]:
    """Fail fast on broken composition instead of discovering missing APIs mid-ingestion."""
    vector_store = getattr(system, "vector_store", None)
    missing = [name for name in REQUIRED_VECTOR_METHODS if not callable(getattr(vector_store, name, None))]
    settings = getattr(system, "settings", None)
    required_dirs = {
        name: getattr(settings, name, None)
        for name in ("incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir", "log_dir")
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


def audit_index_consistency(system: Any, document_id: str | None = None) -> dict[str, Any]:
    """Compare persistent Chroma records against the SQLite lexical mirror."""
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
        metadata = vector_meta[index] if index < len(vector_meta) and isinstance(vector_meta[index], dict) else {}
        vector_map[item_id] = (str(vector_docs[index]) if index < len(vector_docs) else "", dict(metadata))

    with sqlite3.connect(lexical_db) as connection:
        rows = connection.execute("SELECT id, document, metadata, index_state FROM lexical_documents").fetchall()

    lexical_map: dict[str, tuple[str, dict[str, Any], str]] = {}
    for raw_id, document, metadata_json, index_state in rows:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        if document_id and str(metadata.get("document_id")) != str(document_id):
            continue
        lexical_map[str(raw_id)] = (str(document), metadata, str(index_state).upper())

    vector_ready = {
        item_id for item_id, (_, metadata) in vector_map.items()
        if str(metadata.get("index_state", "READY")).upper() == "READY"
    }
    lexical_ready = {item_id for item_id, (_, _, state) in lexical_map.items() if state == "READY"}
    vector_only = sorted(vector_ready - lexical_ready)
    lexical_only = sorted(lexical_ready - vector_ready)
    mismatched: list[str] = []
    for item_id in sorted(vector_ready & lexical_ready):
        vector_doc, vector_metadata = vector_map[item_id]
        lexical_doc, lexical_metadata, _ = lexical_map[item_id]
        if vector_doc != lexical_doc or vector_metadata.get("document_id") != lexical_metadata.get("document_id") or vector_metadata.get("version_id") != lexical_metadata.get("version_id"):
            mismatched.append(item_id)

    return {
        "valid": not vector_only and not lexical_only and not mismatched,
        "vector_count": len(vector_ready),
        "lexical_count": len(lexical_ready),
        "vector_only": vector_only,
        "lexical_only": lexical_only,
        "metadata_or_content_mismatch": mismatched,
    }


def repair_index_consistency(system: Any, document_id: str | None = None) -> dict[str, Any]:
    """Rebuild only the lexical mirror from the authoritative vector records."""
    vector_store = system.vector_store
    audit = audit_index_consistency(system, document_id)
    scope = str(document_id) if document_id else None
    where = {"document_id": document_id} if document_id else None
    records = vector_store.collection.get(where=where, include=["documents", "metadatas"])
    vector_ids = _as_list(records.get("ids"))
    vector_docs = _as_list(records.get("documents"))
    vector_meta = _as_list(records.get("metadatas"))

    expected: dict[str, tuple[str, dict[str, Any]]] = {}
    for index, raw_id in enumerate(vector_ids):
        metadata = vector_meta[index] if index < len(vector_meta) and isinstance(vector_meta[index], dict) else {}
        expected[str(raw_id)] = (str(vector_docs[index]) if index < len(vector_docs) else "", dict(metadata))

    with sqlite3.connect(vector_store.lexical_database) as connection:
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
            connection.executemany("DELETE FROM lexical_documents WHERE id = ?", [(item_id,) for item_id in delete_ids])

    source_docs: list[str] = []
    source_meta: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for item_id, (document, metadata) in expected.items():
        source_ids.append(item_id)
        source_docs.append(document)
        source_meta.append(metadata)
    if source_ids:
        vector_store._upsert_lexical_records(source_docs, source_meta, source_ids)

    final = audit_index_consistency(system, document_id)
    return {
        "repaired": not audit["valid"] and final["valid"],
        "before": audit,
        "after": final,
    }


def run_quality_gate(system: Any, *, repair_drift: bool = True) -> dict[str, Any]:
    """Run deterministic checks and repair recoverable index drift."""
    contract = validate_runtime_contract(system)
    if not contract["ok"]:
        return {"ready": False, "contract": contract, "index": None}

    index = audit_index_consistency(system)
    repair = None
    if repair_drift and not index["valid"]:
        repair = repair_index_consistency(system)
        index = repair["after"]

    return {
        "ready": bool(contract["ok"] and index["valid"]),
        "contract": contract,
        "index": index,
        "repair": repair,
    }
