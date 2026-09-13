from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _matches(meta: dict[str, Any], document_id: str, version_id: str) -> bool:
    if str(meta.get("document_id") or "") != str(document_id):
        return False
    wanted = str(version_id)
    candidates = {
        str(meta.get("version_id") or ""),
        str(meta.get("content_hash") or ""),
    }
    return wanted in candidates


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore

        original = getattr(VectorStore, "delete_version", None)
        if not callable(original) or getattr(original, "_runtime_version_rollback_fix", False):
            _INSTALLED = True
            return

        def delete_version(self: Any, document_id: str, version_id: str) -> None:
            # First use the normal implementation. Then perform an authoritative
            # direct reconciliation because upstream/runtime wrappers may disagree
            # about whether the supplied identifier is version_id or content_hash.
            try:
                original(self, document_id, version_id)
            except Exception:
                pass

            try:
                records = self.collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"],
                )
                removable = []
                for item_id, metadata in zip(
                    records.get("ids", []) or [],
                    records.get("metadatas", []) or [],
                    strict=False,
                ):
                    meta = dict(metadata or {})
                    if _matches(meta, document_id, version_id):
                        removable.append(str(item_id))
                if removable:
                    self.collection.delete(ids=removable)
            except Exception:
                pass

            try:
                with sqlite3.connect(self.lexical_database) as connection:
                    rows = connection.execute(
                        "SELECT id, metadata FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ?",
                        (str(document_id),),
                    ).fetchall()
                    removable = []
                    for row_id, raw in rows:
                        try:
                            meta = json.loads(raw)
                        except Exception:
                            continue
                        if _matches(meta, document_id, version_id):
                            removable.append(str(row_id))
                    if removable:
                        connection.executemany(
                            "DELETE FROM lexical_documents WHERE id = ?",
                            [(item_id,) for item_id in removable],
                        )
                        connection.commit()
            except Exception:
                pass

        delete_version._runtime_version_rollback_fix = True
        delete_version.__name__ = getattr(original, "__name__", "delete_version")
        delete_version.__qualname__ = getattr(original, "__qualname__", "delete_version")
        VectorStore.delete_version = delete_version
        _INSTALLED = True


__all__ = ["install"]
