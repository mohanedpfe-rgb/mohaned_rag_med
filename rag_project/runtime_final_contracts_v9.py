from __future__ import annotations

_INSTALLED = False


def _safe_retire(system, document_id: str, requested_version: str) -> None:
    """Retire only versions proven stale; never delete the current content hash."""
    from pathlib import Path
    import json
    import sqlite3

    store = getattr(system, "vector_store", None)
    if store is None or not document_id:
        return

    allowed_versions: set[str] = {str(requested_version)} if requested_version else set()
    try:
        state = system.state_store.get_document(document_id) or {}
        content_hash = str(state.get("content_hash") or "")
        version_id = str(state.get("version_id") or "")
        if content_hash:
            allowed_versions.add(content_hash)
        if version_id:
            allowed_versions.add(version_id)
    except Exception:
        pass
    allowed_versions.discard("")
    if not allowed_versions:
        return

    database = Path(store.lexical_database)
    try:
        with sqlite3.connect(database, timeout=30) as db:
            db.execute("PRAGMA busy_timeout = 30000")
            rows = db.execute(
                "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                (str(document_id),),
            ).fetchall()
            stale_ids = []
            for item_id, metadata_json in rows:
                try:
                    metadata = json.loads(metadata_json or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
                stored_version = str(metadata.get("version_id") or "")
                if stored_version and stored_version not in allowed_versions:
                    stale_ids.append(str(item_id))
            if stale_ids:
                db.executemany("DELETE FROM lexical_documents WHERE id = ?", [(item_id,) for item_id in stale_ids])
                db.commit()
    except Exception:
        pass

    collection = getattr(store, "collection", None)
    if collection is None:
        return
    try:
        records = collection.get(where={"document_id": document_id}, include=["metadatas"])
        stale_ids = []
        for item_id, metadata in zip(records.get("ids") or [], records.get("metadatas") or [], strict=False):
            stored_version = str((metadata or {}).get("version_id") or "")
            if stored_version and stored_version not in allowed_versions:
                stale_ids.append(str(item_id))
        if stale_ids:
            collection.delete(ids=stale_ids)
    except Exception:
        pass


def install() -> None:
    """Apply the final canonical contract boundary after legacy adapters."""
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.runtime_canonical_contract_guard import install as canonical_guard
    from rag_project.runtime_storage_contract_fix import install as storage_guard
    from rag_project.runtime_test_embedding_guard import install as test_embedding_guard

    canonical_guard()
    storage_guard()
    import rag_project.runtime_storage_contract_fix as storage_fix
    storage_fix._retire = _safe_retire
    test_embedding_guard()
    _INSTALLED = True
