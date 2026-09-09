from __future__ import annotations

# The production answer pipeline is explicitly composed by ProductionRAGSystem.
# This compatibility module intentionally does not monkey-patch RAGSystem.answer.
from rag_project.intelligence.atomic_versioning import install as install_atomic_versioning
from rag_project.intelligence.index_auditor import audit_index

# Existing implementation helpers are imported lazily by ProductionRAGSystem.
# Keep the public feature inventory and report stable for diagnostics.
GOD_MODE_FEATURES = (
    "universal_pdf_routing", "page_quality_scoring", "adaptive_ocr_routing", "alternate_extractor_trigger",
    "ocr_quality_detection", "multi_representation_chunks", "numeric_normalization", "entity_extraction",
    "section_detection", "table_aware_query_planning", "figure_aware_query_planning", "query_intent_classification",
    "query_normalization", "query_decomposition", "query_variants", "document_router", "multi_query_retrieval",
    "parent_child_context_strategy", "neighbor_context_strategy", "metadata_aware_retrieval_boosts", "lexical_vector_fusion",
    "reranking", "evidence_alignment_gate", "evidence_confidence", "context_budget_control", "contextual_compression_hinting",
    "multi_hop_signal", "contradiction_detection", "numeric_consistency_check", "claim_extraction", "claim_level_support_scoring",
    "citation_firewall", "citation_validation", "answer_grounding_gate", "abstention_ladder", "untrusted_evidence_prompt_boundary",
    "prompt_injection_sanitization", "conversation_context_isolation", "query_trace_metadata", "degraded_lexical_fallback",
    "resilient_generation_fallback", "index_visibility_guard", "retrieval_fail_closed_behavior",
)


def report() -> dict[str, object]:
    return {
        "name": "GOD_MODE_RAG",
        "feature_count": len(GOD_MODE_FEATURES),
        "features": list(GOD_MODE_FEATURES),
        "fail_closed": True,
        "universal_pdf_mode": True,
        "answer_monkey_patch": False,
        "composition": "ProductionRAGSystem",
    }


def audit_god_mode_index(system):
    return audit_index(system)


def install() -> None:
    """Install index-version infrastructure only; never patch answer behavior."""
    install_atomic_versioning()
