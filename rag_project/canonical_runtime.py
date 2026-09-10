"""Final runtime binding for the single production answer authority."""
from __future__ import annotations

from typing import Any

ANSWER_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"
CANONICAL_SERVICE = "rag_project.app.production_rag.ProductionRAGSystem"


def install() -> dict[str, Any]:
    """Bind the live production service to the already-installed canonical enhancer.

    ProductionRAGSystem originally captured `enhanced_god_answer` as a class attribute
    during module import.  This binding step makes the runtime reference explicit and
    prevents stale function objects from bypassing later integrity contracts.
    """
    from rag_project.app import production_rag
    from rag_project.intelligence import god_mode_100

    enhancer = god_mode_100.enhance_result
    production_rag.enhanced_god_answer = enhancer
    production_rag.ProductionRAGSystem._certified_god_answer = enhancer
    production_rag.ProductionRAGSystem._canonical_answer_authority = ANSWER_AUTHORITY
    production_rag.ProductionRAGSystem._canonical_runtime_contract = True

    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "class_binding_installed": production_rag.ProductionRAGSystem._certified_god_answer is enhancer,
        "runtime_contract_bound": True,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "install"]
