from __future__ import annotations

import threading

_LOCK = threading.RLock()
_INSTALLED = False


def _retrying_delete_version(original):
    def delete_version(self, document_id: str, version_id: str) -> None:
        matches = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        removable: list[str] = []
        for item_id, metadata in zip(matches.get("ids", []), matches.get("metadatas", []), strict=False):
            meta = self._coerce_metadata(metadata)
            if str(meta.get("version_id") or "") == str(version_id):
                removable.append(str(item_id))

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                if removable:
                    self.collection.delete(ids=removable)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    raise

        with self._sqlite_connection_for_delete() as connection if hasattr(self, "_sqlite_connection_for_delete") else _lexical_connection(self):
            connection.execute(
                "DELETE FROM lexical_documents "
                "WHERE json_extract(metadata, '$.document_id') = ? "
                "AND json_extract(metadata, '$.version_id') = ?",
                (str(document_id), str(version_id)),
            )
            connection.commit()
        if last_error is not None:
            raise last_error

    return delete_version


def _lexical_connection(store):
    import sqlite3
    return sqlite3.connect(store.lexical_database)


def install() -> None:
    """Install the narrow post-publication cleanup contract exactly once."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore
        if not hasattr(VectorStore, "_post_index_v2_original_delete_version"):
            VectorStore._post_index_v2_original_delete_version = VectorStore.delete_version
            VectorStore.delete_version = _retrying_delete_version(VectorStore.delete_version)
        _INSTALLED = True


__all__ = ["install"]
