from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine, RouteMetadata
from rag_project.intelligence.semantic_cache import SemanticRetrievalCache, create_for_system
from rag_project.intelligence.production_contract_v2 import RequestContext


class _ContextRouter:
    """Route adapter that consumes the already-authoritative RequestContext."""

    def __init__(self, context: RequestContext) -> None:
        self.context = context

    def route(self, question: str, context: str, safety: Any) -> RouteMetadata:
        complexity = {"simple": 0.25, "moderate": 0.50, "hard": 0.90}.get(self.context.complexity, 0.50)
        intent = self.context.intent or "factual"
        template = {
            "numeric": "dosage",
            "comparison": "comparison",
            "table": "table",
            "mechanism": "mechanism",
        }.get(intent)
        return RouteMetadata(
            intent=intent,
            complexity=complexity,
            entities=tuple(self.context.entities),
            numeric_sensitivity=bool(self.context.needs_numeric),
            temporal_sensitivity=False,
            conditional_context=(),
            is_follow_up=bool(self.context.is_followup),
            confidence_threshold=float(getattr(safety, "confidence_threshold", 0.75)),
            template_type=template,
            retrieval_timeout_ms=1200 if complexity < 0.65 else 2000,
            needs_multi_hop=bool(self.context.needs_multi_hop),
            query_variants=tuple(self.context.query_variants),
        )


class _InjectedSemanticCache:
    """Compatibility adapter for MedEvidencePro's legacy cache interface."""

    def __init__(self, cache: SemanticRetrievalCache, *, scope_active: bool) -> None:
        self._cache = cache
        self._scope_active = bool(scope_active)

    def get(self, query: str) -> list[dict[str, Any]] | None:
        result = self._cache.get(query, scope_active=self._scope_active)
        if not result:
            return None
        hits, _meta = result
        return [
            {
                "doc_id": hit.doc_id,
                "text": hit.text,
                "metadata": dict(hit.metadata or {}),
                "score": float(hit.score),
                "vector_score": float(hit.vector_score),
                "lexical_score": float(hit.lexical_score),
            }
            for hit in hits
        ]

    def put(self, query: str, hits: Any) -> bool:
        return self._cache.put(query, hits, scope_active=self._scope_active)

    @staticmethod
    def restore(payload: Any):
        return SemanticRetrievalCache.restore(payload)


def execute(
    system: Any,
    question: str,
    metadata_filter: dict[str, Any] | None,
    request_context: RequestContext,
) -> dict[str, Any]:
    """Execute the single answer engine with explicit per-request dependencies."""
    engine = MedEvidenceProEngine(system)
    engine.router = _ContextRouter(request_context)
    engine.retrieval.cache = _InjectedSemanticCache(
        create_for_system(system),
        scope_active=bool(metadata_filter),
    )
    # Keep the request context available for telemetry without making it global.
    engine.request_context = request_context
    return engine.answer(question, metadata_filter)


__all__ = ["execute"]
