from __future__ import annotations

import json
import sqlite3
from typing import Any


def audit_index(vector_store: Any, document_id: str | None = None) -> dict[str, Any]:
    """Detect orphaned, BUILDING, duplicated and cross-store-inconsistent records."""
    where = {"document_id": document_id} if document_id else None
    try:
        records = vector_store.collection.get(where=where, include=["metadatas", "documents"])
        vector_ids = [str(x) for x in records.get("ids", [])]
        vector_meta = [vector_store._coerce_metadata(x) for x in records.get("metadatas", [])]
    except Exception as exc:
        return {"valid": False, "fatal": True, "issues": [f"vector_read_failed: {exc}"]}
    issues: list[str] = []
    building = 0
    logical_seen: set[str] = set()
    for item_id, meta in zip(vector_ids, vector_meta, strict=False):
        state = str(meta.get("index_state", "READY")).upper()
        if state == "BUILDING":
            building += 1
        logical = f"{meta.get('document_id','')}::{meta.get('chunk_id', item_id)}"
        if logical in logical_seen and state == "READY":
            issues.append(f"duplicate_ready_chunk: {logical}")
        logical_seen.add(logical)
        if not meta.get("document_id"):
            issues.append(f"missing_document_id: {item_id}")
        if not meta.get("version_id"):
            issues.append(f"missing_version_id: {item_id}")
    if building:
        issues.append(f"unpublished_building_records: {building}")

    lexical_ids: set[str] = set()
    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            rows = connection.execute("SELECT id, metadata, index_state FROM lexical_documents").fetchall()
        for item_id, raw_meta, state in rows:
            try:
                meta = json.loads(raw_meta)
            except (TypeError, ValueError):
                issues.append(f"invalid_lexical_metadata: {item_id}")
                continue
            if document_id and meta.get("document_id") != document_id:
                continue
            if str(state).upper() == "READY":
                lexical_ids.add(str(item_id))
    except Exception as exc:
        issues.append(f"lexical_read_failed: {exc}")
    ready_vector_ids = {item_id for item_id, meta in zip(vector_ids, vector_meta, strict=False) if str(meta.get("index_state", "READY")).upper() == "READY"}
    only_vector = sorted(ready_vector_ids - lexical_ids)
    only_lexical = sorted(lexical_ids - ready_vector_ids)
    if only_vector:
        issues.append(f"vector_without_lexical: {len(only_vector)}")
    if only_lexical:
        issues.append(f"lexical_without_vector: {len(only_lexical)}")
    return {
        "valid": not issues,
        "fatal": False,
        "document_id": document_id,
        "vector_ready_count": len(ready_vector_ids),
        "lexical_ready_count": len(lexical_ids),
        "building_count": building,
        "issues": issues,
    }
