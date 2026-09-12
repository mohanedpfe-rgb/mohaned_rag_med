from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Any


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _is_chroma_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not isinstance(value, (dict, set, tuple, list))


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata accepted by Chroma without changing meaningful scalar fields.

    Chroma rejects empty list metadata values. It also rejects mappings and nested or
    heterogeneous sequences. Empty optional structural fields are therefore omitted,
    while meaningful structured values are encoded as compact JSON strings when they
    cannot be represented directly by Chroma.
    """
    normalized: dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        if value is None:
            continue
        if _is_chroma_scalar(value):
            normalized[str(key)] = value
            continue
        if isinstance(value, dict):
            if not value:
                continue
            normalized[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            continue
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if not items:
                continue
            if all(_is_chroma_scalar(item) for item in items):
                normalized[str(key)] = items
            else:
                normalized[str(key)] = json.dumps(items, ensure_ascii=False, default=str)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = list(value)
            if not items:
                continue
            if all(_is_chroma_scalar(item) for item in items):
                normalized[str(key)] = items
            else:
                normalized[str(key)] = json.dumps(items, ensure_ascii=False, default=str)
            continue
        normalized[str(key)] = str(value)
    return normalized


def _safe_add_documents(self: Any, documents, metadatas, embeddings, ids):
    normalized = [_normalize_metadata(metadata) for metadata in metadatas]
    return self._chroma_metadata_fix_original_add_documents(
        documents, normalized, embeddings, ids
    )


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore

        if not hasattr(VectorStore, "_chroma_metadata_fix_original_add_documents"):
            VectorStore._chroma_metadata_fix_original_add_documents = VectorStore.add_documents
            VectorStore.add_documents = _safe_add_documents
        _INSTALLED = True


__all__ = ["install"]
