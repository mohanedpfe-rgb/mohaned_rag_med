from __future__ import annotations

from functools import wraps
import json
import sqlite3
from typing import Any


def install() -> None:
    """Final post-index guards: READY-only retrieval and two-sided deletion."""
    from rag_project.storage.vector_store import VectorStore

    original_search = VectorStore.search
    if not getattr(original_search, "_post_index_v2_ready_only", False):

        @wraps(original_search)
        def search(self, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None):
            effective_where = {"index_state": "READY"} if where is None else {"$and": [{"index_state": "READY"}, where]}
            return original_search(self, embedding, n_results=n_results, where=effective_where)

        search._post_index_v2_ready_only = True
        VectorStore.search = search

    original_delete = VectorStore.delete_version
    if getattr(original_delete, "_post_index_v2_two_sided_delete", False):
        return

    @wraps(original_delete)
    def delete_version(self, document_id: str, version_id: str) -> None:
        try:
            return original_delete(self, document_id, version_id)
        except Exception as first_error:
            cleanup_errors: list[Exception] = []

            try:
                matches = self.collection.get(
                    where={"document_id": document_id}, include=["metadatas"]
                )
                removable = []
                for item_id, metadata in zip(
                    matches.get("ids") or [], matches.get("metadatas") or [], strict=False
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

    delete_version._post_index_v2_two_sided_delete = True
    VectorStore.delete_version = delete_version


__all__ = ["install"]
