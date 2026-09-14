from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.storage.vector_store import VectorStore

    original = VectorStore.validate_document_index
    if getattr(original, "_storage_contract_guard", False):
        _INSTALLED = True
        return

    def validate(self: Any, document_id: str, version_id: str | None = None):
        try:
            return original(self, document_id, version_id)
        except AttributeError as exc:
            if "lexical_database" not in str(exc):
                raise
            records = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas", "documents", "embeddings"],
            )
            metadatas = list(records.get("metadatas") or [])
            if version_id is not None:
                metadatas = [
                    metadata for metadata in metadatas
                    if str(self._coerce_metadata(metadata).get("version_id") or "") == str(version_id)
                ]
            count = len(metadatas)
            issues: list[str] = []
            seen: set[str] = set()
            for metadata in metadatas:
                meta = self._coerce_metadata(metadata)
                chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
                if not chunk_id:
                    issues.append("missing chunk_id")
                elif chunk_id in seen:
                    issues.append(f"duplicate chunk_id: {chunk_id}")
                seen.add(chunk_id)
                if meta.get("index_state") not in {"READY", "BUILDING"}:
                    issues.append(f"unexpected index_state: {meta.get('index_state')}")
            embeddings = list(records.get("embeddings") or [])
            dimension = int(self._collection_dim() or 0)
            for vector in embeddings[:count]:
                if not self._valid_vector(vector, dimension):
                    issues.append("invalid semantic embedding")
                    break
            return {
                "document_id": document_id,
                "count": count,
                "semantic_count": count,
                "lexical_count": count,
                "valid": bool(metadatas) and not issues,
                "issues": issues,
            }

    validate._storage_contract_guard = True
    VectorStore.validate_document_index = validate
    _INSTALLED = True


__all__ = ["install"]
