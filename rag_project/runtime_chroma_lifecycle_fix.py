from __future__ import annotations

import gc
import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _clear_cache() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        clear_system_cache = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear_system_cache):
            clear_system_cache()
    except Exception:
        pass


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore

        if getattr(VectorStore, "close", None) is None:
            def close(self: Any) -> None:
                # Drop the per-store Chroma object graph before clearing the
                # process-level shared cache. This releases HNSW mmap/file
                # handles on Windows before TemporaryDirectory cleanup.
                collection = getattr(self, "collection", None)
                client = getattr(self, "client", None)
                self.collection = None
                self.client = None
                self.expected_identity = None
                del collection, client
                gc.collect()
                _clear_cache()
                gc.collect()
                _clear_cache()

            VectorStore.close = close
        _INSTALLED = True


__all__ = ["install"]
