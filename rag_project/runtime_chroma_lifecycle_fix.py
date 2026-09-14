from __future__ import annotations

import gc
import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _stop_legacy_client(client: Any) -> None:
    """Best-effort shutdown for Chroma versions without Client.close().

    Older Chroma releases can retain native HNSW handles through
    LocalSegmentManager. Release the segment implementations and caches before
    dropping the Python references to the client.
    """
    if client is None:
        return

    server = getattr(client, "_server", None)
    manager = getattr(server, "_manager", None)
    if manager is not None:
        instances = getattr(manager, "_instances", None)
        if isinstance(instances, dict):
            for instance in list(instances.values()):
                stop = getattr(instance, "stop", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        pass
            try:
                instances.clear()
            except Exception:
                pass

        caches = getattr(manager, "segment_cache", None)
        if isinstance(caches, dict):
            for cache in list(caches.values()):
                reset = getattr(cache, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass

    system = getattr(client, "_system", None)
    if system is None:
        system = getattr(server, "_system", None)
    system_stop = getattr(system, "stop", None)
    if callable(system_stop):
        try:
            system_stop()
        except Exception:
            pass

    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        clear_system_cache = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear_system_cache):
            clear_system_cache()
    except Exception:
        pass


def _close_client(client: Any) -> None:
    """Close the Chroma client that owns the native HNSW resources."""
    if client is None:
        return

    client_close = getattr(client, "close", None)
    if callable(client_close):
        client_close()
        return

    _stop_legacy_client(client)


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        from rag_project.storage.vector_store import VectorStore

        def close(self: Any) -> None:
            if getattr(self, "_chroma_closed", False):
                return
            client = getattr(self, "client", None)
            try:
                _close_client(client)
            finally:
                self.collection = None
                self.client = None
                self.expected_identity = None
                self._chroma_closed = True
                gc.collect()

        close._chroma_lifecycle_fix = True
        VectorStore.close = close
        VectorStore.__enter__ = lambda self: self
        VectorStore.__exit__ = lambda self, exc_type, exc_value, traceback: self.close()
        _INSTALLED = True


__all__ = ["install"]
