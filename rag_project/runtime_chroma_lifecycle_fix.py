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
                collection = getattr(self, "collection", None)
                client = getattr(self, "client", None)
                try:
                    # Chroma's PersistentClient now exposes an explicit close()
                    # because GC alone is insufficient to release SQLite/HNSW
                    # resources on Windows. Always call it before dropping refs.
                    client_close = getattr(client, "close", None)
                    if callable(client_close):
                        client_close()
                except Exception:
                    pass
                self.collection = None
                self.client = None
                self.expected_identity = None
                del collection, client
                gc.collect()
                _clear_cache()
                gc.collect()
                _clear_cache()

            close._chroma_lifecycle_fix = True
            VectorStore.close = close
        else:
            original_close = VectorStore.close
            if not getattr(original_close, "_chroma_lifecycle_fix_wrapped", False):
                def wrapped_close(self: Any) -> None:
                    client = getattr(self, "client", None)
                    try:
                        client_close = getattr(client, "close", None)
                        if callable(client_close):
                            client_close()
                    except Exception:
                        pass
                    try:
                        original_close(self)
                    finally:
                        _clear_cache()
                        gc.collect()
                        _clear_cache()

                wrapped_close._chroma_lifecycle_fix_wrapped = True
                VectorStore.close = wrapped_close
        _INSTALLED = True


__all__ = ["install"]
