from __future__ import annotations

import re
from dataclasses import replace
from typing import Any


_NUMERIC_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b",
    re.I,
)


class ReadyOnlyRetriever:
    """Retrieval boundary that makes READY index visibility mandatory.

    Ingestion may write vector records while a document is RUNNING/BUILDING.
    Search must never expose those records, regardless of caller-supplied filters.

    The same boundary also applies deterministic evidence-aware reranking so
    lexical tier-0 retrieval cannot bypass ranking policy used by normal hybrid
    retrieval.
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

    @staticmethod
    def _score_bonus(query: str, text: str, metadata: dict[str, Any]) -> float:
        lowered_query = str(query or "").casefold()
        lowered_text = str(text or "").casefold()
        bonus = 0.0

        tokens = {token for token in re.findall(r"\w+", lowered_query, flags=re.UNICODE) if len(token) >= 3}
        token_hits = sum(1 for token in tokens if token in lowered_text)
        bonus += min(0.18, 0.025 * token_hits)

        asks_table = any(term in lowered_query for term in ("table", "tableau", "rows", "columns", "جدول"))
        is_table = (
            "table" in lowered_text
            or str(metadata.get("representation_type", "")).casefold() == "table"
            or bool(metadata.get("table_id"))
        )
        if asks_table and is_table:
            bonus += 0.40

        asks_numeric = bool(
            _NUMERIC_PATTERN.search(lowered_query)
            or any(term in lowered_query for term in ("dose", "dosage", "how much", "how many", "value", "range", "جرعة", "قيمة"))
        )
        if asks_numeric and _NUMERIC_PATTERN.search(lowered_text):
            bonus += 0.28

        for phrase in ("hba1c", "glycemic control", "metformin", "diabetes mellitus"):
            if phrase in lowered_query and phrase in lowered_text:
                bonus += 0.12
        return min(0.90, bonus)

    @classmethod
    def _rerank_raw(cls, lexical_query: str, result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        ids_value = result.get("ids")
        documents_value = result.get("documents")
        metadatas_value = result.get("metadatas")
        distances_value = result.get("distances")
        ids = ids_value[0] if isinstance(ids_value, list) and ids_value and isinstance(ids_value[0], list) else ids_value or []
        documents = documents_value[0] if isinstance(documents_value, list) and documents_value and isinstance(documents_value[0], list) else documents_value or []
        metadatas = metadatas_value[0] if isinstance(metadatas_value, list) and metadatas_value and isinstance(metadatas_value[0], list) else metadatas_value or []
        distances = distances_value[0] if isinstance(distances_value, list) and distances_value and isinstance(distances_value[0], list) else distances_value or []
        n = min(len(ids), len(documents), len(metadatas) if metadatas else len(ids))
        if n <= 1:
            return result
        rows = []
        for index in range(n):
            metadata = dict(metadatas[index]) if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            text = str(documents[index] or "")
            distance = float(distances[index]) if index < len(distances) else 1e9
            base = 1.0 / (1.0 + max(0.0, distance))
            bonus = cls._score_bonus(lexical_query, text, metadata)
            rows.append((base + bonus, index, bonus))
        rows.sort(key=lambda item: item[0], reverse=True)

        def reorder(values: Any) -> Any:
            if isinstance(values, list) and values and isinstance(values[0], list):
                inner = values[0]
                return [[inner[index] for _, index, _ in rows]]
            return [values[index] for _, index, _ in rows] if isinstance(values, list) else values

        output = dict(result)
        output["ids"] = reorder(ids_value)
        output["documents"] = reorder(documents_value)
        output["metadatas"] = reorder(metadatas_value)
        if distances_value is not None:
            output["distances"] = reorder(distances_value)
        return output

    @classmethod
    def _rerank_hits(cls, query: str, hits: Any):
        if not hits:
            return hits
        rows = []
        for index, hit in enumerate(hits):
            metadata = dict(getattr(hit, "metadata", {}) or {})
            text = str(getattr(hit, "text", "") or "")
            base = float(getattr(hit, "score", 0.0) or 0.0)
            bonus = cls._score_bonus(query, text, metadata)
            try:
                ranked = replace(hit, score=base + bonus, metadata={**metadata, "ranking_score_base": round(base, 6), "intent_rerank_bonus": round(bonus, 6), "ranking_score": round(base + bonus, 6)})
            except TypeError:
                hit.score = base + bonus
                hit.metadata = {**metadata, "ranking_score_base": round(base, 6), "intent_rerank_bonus": round(bonus, 6), "ranking_score": round(base + bonus, 6)}
                ranked = hit
            rows.append((float(getattr(ranked, "score", 0.0)), index, ranked))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [item[2] for item in rows]

    def retrieve(self, query: str, top_k: int = 6, where: dict[str, Any] | None = None):
        hits = self._retriever.retrieve(query, top_k, self._ready_where(where))
        return self._rerank_hits(query, hits)[:top_k]

    def _lexical(self, lexical_query: str, candidate_count: int, where: dict[str, Any] | None):
        raw = self._retriever._lexical(lexical_query, candidate_count, self._ready_where(where))
        return self._rerank_raw(lexical_query, raw)

    def _vector(self, query_embedding: Any, candidate_count: int, where: dict[str, Any] | None):
        return self._retriever._vector(query_embedding, candidate_count, self._ready_where(where))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)


__all__ = ["ReadyOnlyRetriever"]