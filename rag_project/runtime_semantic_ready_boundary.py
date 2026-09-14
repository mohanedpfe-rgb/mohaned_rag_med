from __future__ import annotations

from functools import wraps
from typing import Any


def install() -> None:
    """Prevent semantic retrieval from returning non-READY chunks."""
    from rag_project.storage.vector_store import VectorStore

    original_search = VectorStore.search
    if getattr(original_search, "_semantic_ready_boundary", False):
        return

    @wraps(original_search)
    def search(self, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None):
        if where is None:
            effective_where = {"index_state": "READY"}
        else:
            effective_where = {"$and": [{"index_state": "READY"}, where]}
        return original_search(self, embedding, n_results=n_results, where=effective_where)

    search._semantic_ready_boundary = True
    VectorStore.search = search


__all__ = ["install"]
