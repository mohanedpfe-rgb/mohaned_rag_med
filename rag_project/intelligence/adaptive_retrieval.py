from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RetrievalBudget:
    variant_limit: int
    candidate_limit: int
    rerank_limit: int
    depth: int
    retry: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_retrieval_budget(*, query_tokens: int, entity_count: int, intent: str, confidence: float, initial_score: float | None = None) -> RetrievalBudget:
    hard_intents = {"comparison", "diagnosis", "management", "etiology", "mechanism", "prognosis", "relationship", "numeric"}
    hard = intent in hard_intents or entity_count >= 3 or query_tokens >= 18
    weak = initial_score is not None and initial_score < 0.35
    if weak:
        return RetrievalBudget(12, 72, 64, 3, True, "weak_initial_alignment")
    if hard and confidence < 0.88:
        return RetrievalBudget(10, 60, 56, 3, False, "hard_query_low_confidence")
    if hard:
        return RetrievalBudget(8, 48, 48, 2, False, "hard_query")
    return RetrievalBudget(4, 24, 24, 1, False, "simple_query")


def should_retry_retrieval(*, alignment: float, entity_coverage: float, contradiction: float, attempts: int = 0, max_attempts: int = 1) -> bool:
    if attempts >= max_attempts:
        return False
    if contradiction >= 0.75:
        return True
    return alignment < 0.30 or entity_coverage < 0.50
