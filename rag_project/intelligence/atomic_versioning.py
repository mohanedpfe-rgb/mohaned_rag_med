from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any, Sequence


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _stamp_metadata(metadata: dict[str, Any], physical_id: str) -> dict[str, Any]:
    meta = dict(metadata or {})
    meta.setdefault("version_id", meta.get("document_id", "legacy"))
    meta["storage_id"] = physical_id
    meta["build_id"] = str(meta.get("build_id") or uuid.uuid4().hex)
    meta["index_state"] = str(meta.get("index_state", "BUILDING")).upper()
    # Preserve the logical chunk id for citations/retrieval while making the physical
    # storage id unique across simultaneous rebuilds of the same content.
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
    """Promote only the current BUILDING generation; never destroy a READY generation first."""
    desired = str(state).upper()
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    build_ids: set[str] = set()
    old_ready_ids: list[str] = []
    current_build: str | None = None
    for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
        meta = self._coerce_metadata(metadata)
        if meta.get("version_id") != version_id:
            continue
        if str(meta.get("index_state", "READY")).upper() == "BUILDING":
            current_build = current_build or str(meta.get("build_id", ""))
            build_ids.add(str(item_id))
    if desired == "READY" and not build_ids:
        # Backward-compatible behavior for legacy indexes that have no build_id.
        return self._god_atomic_original_set_version_index_state(document_id, version_id, desired)
    if build_ids:
        promote_meta = []
        promote_ids = []
        for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
            if str(item_id) not in build_ids:
                continue
            meta = self._coerce_metadata(metadata)
            meta["index_state"] = desired
            promote_ids.append(str(item_id))
            promote_meta.append(meta)
        if promote_ids:
            self.collection.update(ids=promote_ids, metadatas=promote_meta)
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
            matching = []
            for item_id, raw in rows:
                try:
                    meta = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if meta.get("document_id") == document_id and meta.get("version_id") == version_id and meta.get("build_id") in {current_build}:
                    matching.append(str(item_id))
            if matching:
                connection.executemany(
                    "UPDATE lexical_documents SET index_state=?, metadata=json_set(metadata, '$.index_state', ?) WHERE id=?",
                    [(desired, desired, item_id) for item_id in matching],
                )
        if desired == "READY":
            # Same-content/config rebuild: after the new generation is published, remove
            # older READY generations with this exact version id. A failed build never reaches here.
            all_ids = []
            for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
                meta = self._coerce_metadata(metadata)
                if meta.get("version_id") == version_id and str(meta.get("index_state", "READY")).upper() == "READY" and str(meta.get("build_id", "")) != str(current_build):
                    all_ids.append(str(item_id))
            if all_ids:
                self.collection.delete(ids=all_ids)
                with sqlite3.connect(self.lexical_database) as connection:
                    connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id in all_ids])


def _safe_delete_version(self: Any, document_id: str, version_id: str) -> None:
    """Delete only uncommitted generations for a version; READY data is immutable until replacement is READY."""
    records = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
    deleting: list[str] = []
    builds: set[str] = set()
    for item_id, metadata in zip(records.get("ids", []), records.get("metadatas", []), strict=False):
        meta = self._coerce_metadata(metadata)
        if meta.get("version_id") == version_id and str(meta.get("index_state", "READY")).upper() == "BUILDING":
            deleting.append(str(item_id))
            builds.add(str(meta.get("build_id", "")))
    if deleting:
        self.collection.delete(ids=deleting)
        with sqlite3.connect(self.lexical_database) as connection:
            rows = connection.execute("SELECT id, metadata FROM lexical_documents").fetchall()
            ids = []
            for item_id, raw in rows:
                try:
                    meta = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if meta.get("document_id") == document_id and meta.get("version_id") == version_id and meta.get("build_id") in builds:
                    ids.append(str(item_id))
            if ids:
                connection.executemany("DELETE FROM lexical_documents WHERE id=?", [(item_id,) for item_id in ids])
    elif not deleting:
        # Legacy callers can explicitly delete a version only when it is not READY.
        # READY records are intentionally protected here to preserve the last known-good index.
        return


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
