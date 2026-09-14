from __future__ import annotations

from functools import wraps
from typing import Any


def install() -> None:
    """READY-only semantic retrieval guard used after the legacy contract stack."""
    from rag_project.storage.vector_store import VectorStore
    original = VectorStore.search
    if getattr(original, "_post_index_v2_ready_only", False):
        return

    @wraps(original)
    def search(self, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None):
        effective_where = {"index_state": "READY"} if where is None else {"$and": [{"index_state": "READY"}, where]}
        return original(self, embedding, n_results=n_results, where=effective_where)

    search._post_index_v2_ready_only = True
    VectorStore.search = search


__all__ = ["install"]
