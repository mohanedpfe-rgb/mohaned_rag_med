from __future__ import annotations

import threading

_LOCK = threading.RLock()
_INSTALLED = False


def _retrying_delete_version(original):
    def delete_version(self, document_id: str, version_id: str) -> None:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                original(self, document_id, version_id)
                return
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    raise
        if last_error is not None:
            raise last_error

    delete_version.__name__ = getattr(original, "__name__", "delete_version")
    delete_version.__qualname__ = getattr(original, "__qualname__", "delete_version")
    return delete_version


def install() -> None:
    """Install a narrow retry boundary without replacing deletion semantics."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore
        if not hasattr(VectorStore, "_post_index_v2_original_delete_version"):
            original = VectorStore.delete_version
            VectorStore._post_index_v2_original_delete_version = original
            VectorStore.delete_version = _retrying_delete_version(original)
        _INSTALLED = True


__all__ = ["install"]
