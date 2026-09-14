from __future__ import annotations

from functools import wraps
from typing import Any


def _sanitize_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    normalized = dict(metadata)
    # Chroma 1.5.x rejects empty list metadata values. Missing optional page
    # metadata is represented by omission; real ingested chunks retain their
    # non-empty page_numbers list.
    for key, value in list(normalized.items()):
        if isinstance(value, list) and not value:
            normalized.pop(key, None)
    return normalized


def install() -> None:
    from rag_project.storage.vector_store import VectorStore

    current = VectorStore._coerce_metadata
    if getattr(current, "_empty_list_sanitizer", False):
        return

    @wraps(current)
    def wrapped(self: Any, metadata: Any) -> dict[str, Any]:
        return _sanitize_metadata(current(self, metadata))

    wrapped._empty_list_sanitizer = True
    VectorStore._coerce_metadata = wrapped


__all__ = ["install"]
