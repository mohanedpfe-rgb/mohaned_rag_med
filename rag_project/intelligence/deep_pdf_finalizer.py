from __future__ import annotations

import threading

from rag_project.storage.vector_store import VectorStore

_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        # A legacy atomic-versioning installer may wrap the deep contract after it
        # is installed. Restore the deep contract as the final owner of these
        # methods while preserving the original references for diagnostics.
        for public_name, original_name in (
            ("add_documents", "_god_atomic_original_add_documents"),
            ("add_lexical_documents", "_god_atomic_original_add_lexical_documents"),
            ("set_version_index_state", "_god_atomic_original_set_version_index_state"),
            ("delete_version", "_god_atomic_original_delete_version"),
        ):
            original = getattr(VectorStore, original_name, None)
            if original is not None and getattr(VectorStore, public_name, None) is not original:
                setattr(VectorStore, public_name, original)
        _INSTALLED = True


__all__ = ["install"]
