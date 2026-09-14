from __future__ import annotations

from functools import wraps
import json
import sqlite3
from typing import Any


def install() -> None:
    """Ensure semantic and lexical version deletion are both attempted on partial failure."""
    from rag_project.storage.vector_store import VectorStore

    original = VectorStore.delete_version
    if getattr(original, "_two_sided_delete_consistency", False):
        return

    @wraps(original)
    def delete_version(self, document_id: str, version_id: str) -> None:
        try:
            return original(self, document_id, version_id)
        except Exception as first_error:
            cleanup_errors: list[Exception] = []
            try:
                matches = self.collection.get(
                    where={"document_id": document_id}, include=["metadatas"]
                )
                removable = []
                for item_id, metadata in zip(
                    (matches.get("ids") or []), (matches.get("metadatas") or []), strict=False
                ):
                    if isinstance(metadata, dict) and str(metadata.get("version_id")) == str(version_id):
                        removable.append(str(item_id))
                if removable:
                    self.collection.delete(ids=removable)
            except Exception as exc:
                cleanup_errors.append(exc)

            try:
                with sqlite3.connect(self.lexical_database) as connection:
                    rows = connection.execute(
                        "SELECT id, metadata FROM lexical_documents "
                        "WHERE json_extract(metadata, '$.document_id') = ?",
                        (str(document_id),),
                    ).fetchall()
                    matching = []
                    for row_id, raw_metadata in rows:
                        try:
                            metadata = json.loads(raw_metadata or "{}")
                        except (TypeError, ValueError, json.JSONDecodeError):
                            metadata = {}
                        if str(metadata.get("version_id")) == str(version_id):
                            matching.append(str(row_id))
                    if matching:
                        connection.executemany(
                            "DELETE FROM lexical_documents WHERE id = ?",
                            [(item_id,) for item_id in matching],
                        )
                        connection.commit()
            except Exception as exc:
                cleanup_errors.append(exc)

            if cleanup_errors:
                raise RuntimeError(
                    f"Version deletion failed and cleanup was incomplete after {type(first_error).__name__}: "
                    + ", ".join(type(exc).__name__ for exc in cleanup_errors)
                ) from first_error
            raise

    delete_version._two_sided_delete_consistency = True
    VectorStore.delete_version = delete_version


__all__ = ["install"]
