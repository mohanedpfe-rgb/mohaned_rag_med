from __future__ import annotations

from functools import wraps
from typing import Any


def install() -> None:
    """Final semantic retrieval publication fence."""
    from rag_project.storage.vector_store import VectorStore
    original = VectorStore.search
    if getattr(original, "_semantic_ready_v2", False):
        return
    @wraps(original)
    def search(self, embedding: Any, n_results: int = 5, where: dict[str, Any] | None = None):
        effective = {"index_state": "READY"} if where is None else {"$and": [{"index_state": "READY"}, where]}
        return original(self, embedding, n_results=n_results, where=effective)
    search._semantic_ready_v2 = True
    VectorStore.search = search
