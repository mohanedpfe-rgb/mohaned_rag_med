from __future__ import annotations

import gc
import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _clear_cache() -> None:
    """Do not clear Chroma's process-wide client registry from one store's close().

    Multiple isolated VectorStore instances may share the Chroma registry during
    pytest-xdist/threaded tests. Clearing that global registry here invalidates
    still-live clients and can surface as Collection-not-found on another store.
    """
    return None


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.storage.vector_store import VectorStore

        if getattr(VectorStore, "close", None) is None:
            def close(self: Any) -> None:
                client = getattr(self, "client", None)
                try:
                    client_close = getattr(client, "close", None)
                    if callable(client_close):
                        client_close()
                except Exception:
                    pass
                self.collection = None
                self.client = None
                self.expected_identity = None
                gc.collect()

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
                        gc.collect()

                wrapped_close._chroma_lifecycle_fix_wrapped = True
                VectorStore.close = wrapped_close
        _INSTALLED = True


__all__ = ["install"]
