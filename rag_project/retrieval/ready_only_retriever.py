from __future__ import annotations

from typing import Any


class ReadyOnlyRetriever:
    """Retrieval boundary that makes READY index visibility mandatory.

    Ingestion may write vector records while a document is RUNNING/BUILDING.
    Search must never expose those records, regardless of caller-supplied filters.
    """

    REQUIRED_STATE = "READY"

    def __init__(self, retriever: Any):
        if retriever is None:
            raise ValueError("A concrete retriever is required.")
        self._retriever = retriever

    @classmethod
    def _ready_where(cls, where: dict[str, Any] | None) -> dict[str, Any]:
        required = {"index_state": cls.REQUIRED_STATE}
        if not where:
            return required
        # Never permit a caller to weaken the READY-only boundary.
        caller = dict(where)
        if caller.get("index_state") not in (None, cls.REQUIRED_STATE):
            caller.pop("index_state", None)
        if not caller:
            return required
        return {"$and": [required, caller]}

    def retrieve(self, query: str, top_k: int = 6, where: dict[str, Any] | None = None):
        return self._retriever.retrieve(query, top_k, self._ready_where(where))

    def _lexical(self, lexical_query: str, candidate_count: int, where: dict[str, Any] | None):
        return self._retriever._lexical(lexical_query, candidate_count, self._ready_where(where))

    def _vector(self, query_embedding: Any, candidate_count: int, where: dict[str, Any] | None):
        return self._retriever._vector(query_embedding, candidate_count, self._ready_where(where))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)


__all__ = ["ReadyOnlyRetriever"]
