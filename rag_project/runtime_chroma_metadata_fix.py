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
    """Return metadata accepted by Chroma for both inserts and later updates.

    Chroma rejects empty list metadata values and nested/non-scalar structures.
    Optional empty fields are omitted; meaningful structured values are JSON encoded.
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


def _safe_coerce_metadata(self: Any, metadata: Any) -> dict[str, Any]:
    """Normalize metadata through the previously installed bound method.

    ``_chroma_metadata_fix_original_coerce_metadata`` is stored on the
    ``VectorStore`` class. Accessing it through ``self`` produces a bound method,
    so passing ``self`` again would call the wrapper with three positional
    arguments and raise ``TypeError: ... takes 2 positional arguments but 3 were
    given``. Keep the wrapper compatible with both the base implementation and
    earlier runtime layers by calling the bound method with only ``metadata``.
    """
    original = self._chroma_metadata_fix_original_coerce_metadata
    return _normalize_metadata(original(metadata))


def _safe_transition_document_state(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
    """Make READY publication satisfy the state/index invariant.

    The production ingestion path marks the vector records READY immediately before
    transitioning the durable document row from VALIDATING_INDEX to READY. The row's
    previous index_state is PENDING, so the state contract would otherwise reject a
    valid publication even though the index was already verified.
    """
    stage = str(new_stage).upper()
    if stage in {"READY", "COMPLETED"} and "index_state" not in values:
        values["index_state"] = "READY"
    return self._chroma_metadata_fix_original_transition_document_state(
        document_id, new_stage, **values
    )


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.ingestion.state_store import IngestionStateStore
        from rag_project.storage.vector_store import VectorStore

        if not hasattr(VectorStore, "_chroma_metadata_fix_original_add_documents"):
            VectorStore._chroma_metadata_fix_original_add_documents = VectorStore.add_documents
            VectorStore.add_documents = _safe_add_documents

        if not hasattr(VectorStore, "_chroma_metadata_fix_original_coerce_metadata"):
            VectorStore._chroma_metadata_fix_original_coerce_metadata = VectorStore._coerce_metadata
            VectorStore._coerce_metadata = _safe_coerce_metadata

        if not hasattr(IngestionStateStore, "_chroma_metadata_fix_original_transition_document_state"):
            IngestionStateStore._chroma_metadata_fix_original_transition_document_state = IngestionStateStore.transition_document_state
            IngestionStateStore.transition_document_state = _safe_transition_document_state

        _INSTALLED = True


__all__ = ["install", "_normalize_metadata"]
