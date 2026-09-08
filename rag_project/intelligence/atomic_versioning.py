from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any, Sequence

from rag_project.intelligence.pdf_intelligence import enrich_text


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _stamp_metadata(metadata: dict[str, Any], physical_id: str) -> dict[str, Any]:
    meta = dict(metadata or {})
    meta.setdefault("version_id", meta.get("document_id", "legacy"))
    meta["storage_id"] = physical_id
    meta["build_id"] = str(meta.get("build_id") or uuid.uuid4().hex)
    meta["index_state"] = str(meta.get("index_state", "BUILDING")).upper()
    try:
        enriched = enrich_text(str(meta.get("source_text", ""))) if meta.get("source_text") else {}
        if enriched:
            meta["keywords"] = json.dumps(enriched.get("keywords", []), ensure_ascii=False)
            meta["entities"] = json.dumps(enriched.get("entities", {}), ensure_ascii=False, sort_keys=True)
            meta["normalized_numbers"] = json.dumps(enriched.get("number_forms", []), ensure_ascii=False)
    except Exception:
        # Metadata enrichment is auxiliary; it must never block an otherwise valid index write.
        pass
    meta.pop("source_text", None)
    return meta


def _safe_add(self: Any, documents: Sequence[str], metadatas: Sequence[dict[str, Any]], embeddings: Sequence[Sequence[float]], ids: Sequence[str]):
    build_id = uuid.uuid4().hex
    physical_ids: list[str] = []
    stamped: list[dict[str, Any]] = []
    for index, logical_id in enumerate(ids):
        physical = f"{logical_id}-build-{build_id[:16]}-{index}"
        physical_ids.append(physical)
        meta = dict(metadatas[index] or {})
        meta["build_id"] = build_id
        meta = _stamp_metadata(meta, physical)
        stamped.append(meta)
    return self._god_atomic_original_add_documents(documents, stamped, embeddings, physical_ids)


def _safe_add_lexical(self: Any, documents: Sequence[str], metadatas: Sequence[dict[str, Any]], ids: Sequence[str]):
    build_id = uuid.uuid4().hex
    physical_ids: list[str] = []
    stamped: list[dict[str, Any]] = []
    for index, logical_id in enumerate(ids):
        physical = f"{logical_id}-build-{build_id[:16]}-{index}"
        physical_ids.append(physical)
        meta = dict(metadatas[index] or {})
        meta["build_id"] = build_id
        meta = _stamp_metadata(meta, physical)
        stamped.append(meta)
    return self._god_atomic_original_add_lexical_documents(documents, stamped, physical_ids)


def _safe_set_version_state(self: Any, document_id: str, version_id: str, state: str) -> None:
    """Publish all BUILDING batches for the requested version as one logical generation."""
    desired = str(state).upper()
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    build_ids: set[str] = set()
    building_records: list[tuple[str, dict[str, Any]]] = []
    for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
        meta = self._coerce_metadata(metadata)
        if meta.get("version_id") == version_id and str(meta.get("index_state", "READY")).upper() == "BUILDING":
            build_id = str(meta.get("build_id", ""))
            build_ids.add(build_id)
            building_records.append((str(item_id), meta))
    if desired == "READY" and not building_records:
        return self._god_atomic_original_set_version_index_state(document_id, version_id, desired)
    if not building_records:
        return
    ids_to_promote = [item_id for item_id, _ in building_records]
    metas_to_promote = []
    for _, meta in building_records:
        meta["index_state"] = desired
        metas_to_promote.append(meta)
    self.collection.update(ids=ids_to_promote, metadatas=metas_to_promote)
    with sqlite3.connect(self.lexical_database) as connection:
        rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
        matching = []
        for item_id, raw in rows:
            try:
                meta = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if meta.get("document_id") == document_id and meta.get("version_id") == version_id and str(meta.get("build_id", "")) in build_ids:
                matching.append(str(item_id))
        if matching:
            connection.executemany(
                "UPDATE lexical_documents SET index_state=?, metadata=json_set(metadata, '$.index_state', ?) WHERE id=?",
                [(desired, desired, item_id) for item_id in matching],
            )
    if desired == "READY":
        # Remove older generations only AFTER every new batch was promoted.
        old_ids = []
        for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
            meta = self._coerce_metadata(metadata)
            if meta.get("version_id") == version_id and str(meta.get("index_state", "READY")).upper() == "READY" and str(meta.get("build_id", "")) not in build_ids:
                old_ids.append(str(item_id))
        if old_ids:
            self.collection.delete(ids=old_ids)
            with sqlite3.connect(self.lexical_database) as connection:
                connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id in old_ids])


def _safe_delete_version(self: Any, document_id: str, version_id: str) -> None:
    """Roll back uncommitted data only. Last-known-good READY generations survive."""
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    deleting: list[str] = []
    builds: set[str] = set()
    for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
        meta = self._coerce_metadata(metadata)
        if meta.get("version_id") == version_id and str(meta.get("index_state", "READY")).upper() == "BUILDING":
            deleting.append(str(item_id))
            builds.add(str(meta.get("build_id", "")))
    if not deleting:
        return
    self.collection.delete(ids=deleting)
    with sqlite3.connect(self.lexical_database) as connection:
        rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
        ids = []
        for item_id, raw in rows:
            try:
                meta = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if meta.get("document_id") == document_id and meta.get("version_id") == version_id and str(meta.get("build_id", "")) in builds:
                ids.append(str(item_id))
        if ids:
            connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id in ids])


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore
        if not hasattr(VectorStore, "_god_atomic_original_add_documents"):
            VectorStore._god_atomic_original_add_documents = VectorStore.add_documents
            VectorStore.add_documents = _safe_add
        if hasattr(VectorStore, "add_lexical_documents") and not hasattr(VectorStore, "_god_atomic_original_add_lexical_documents"):
            VectorStore._god_atomic_original_add_lexical_documents = VectorStore.add_lexical_documents
            VectorStore.add_lexical_documents = _safe_add_lexical
        if not hasattr(VectorStore, "_god_atomic_original_set_version_index_state"):
            VectorStore._god_atomic_original_set_version_index_state = VectorStore.set_version_index_state
            VectorStore.set_version_index_state = _safe_set_version_state
        if not hasattr(VectorStore, "_god_atomic_original_delete_version"):
            VectorStore._god_atomic_original_delete_version = VectorStore.delete_version
            VectorStore.delete_version = _safe_delete_version
        _INSTALLED = True
