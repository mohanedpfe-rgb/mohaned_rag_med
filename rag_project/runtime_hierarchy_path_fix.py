from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _hierarchy_path(metadata: Mapping[str, Any]) -> str:
    existing = metadata.get("hierarchy_path")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    if isinstance(existing, (list, tuple)):
        parts = [str(value).strip() for value in existing if str(value).strip()]
        if parts:
            return " > ".join(parts)

    chapter = str(metadata.get("chapter") or "").strip()
    section = str(metadata.get("section") or "").strip()
    parent = str(metadata.get("parent_id") or "").strip()
    document = str(metadata.get("document_id") or "").strip()

    parts = [value for value in (chapter, section) if value]
    if not parts and parent:
        parts = [parent]
    if not parts:
        parts = [f"document:{document}" if document else "document:unknown"]
    return " > ".join(parts)


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    row = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
    row["hierarchy_path"] = _hierarchy_path(row)
    return row


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore

        current = getattr(VectorStore, "add_documents", None)
        if not callable(current) or getattr(current, "_runtime_hierarchy_path_fix", False):
            _INSTALLED = True
            return

        def add_documents(self: Any, documents, metadatas, embeddings, ids):
            normalized = [_normalize_metadata(metadata) for metadata in (metadatas or [])]
            return current(self, documents, normalized, embeddings, ids)

        add_documents.__name__ = getattr(current, "__name__", "add_documents")
        add_documents.__qualname__ = getattr(current, "__qualname__", add_documents.__name__)
        add_documents._runtime_hierarchy_path_fix = True
        VectorStore.add_documents = add_documents
        _INSTALLED = True


__all__ = ["install"]