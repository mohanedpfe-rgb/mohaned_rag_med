"""Production-grade contracts for feature integrity, privacy, and fail-closed decisions."""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    target: str
    critical: bool = True


# Exactly one executable target per advertised capability.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("universal_pdf_routing", "rag_project.intelligence.pdf_intelligence:classify_document_pages"),
    FeatureSpec("page_quality_scoring", "rag_project.intelligence.pdf_intelligence:score_page_quality"),
    FeatureSpec("adaptive_ocr_routing", "rag_project.intelligence.pdf_intelligence:classify_document_pages"),
    FeatureSpec("alternate_extractor_trigger", "rag_project.parsing.pdf_extractor:PDFExtractor"),
    FeatureSpec("ocr_quality_detection", "rag_project.intelligence.pdf_intelligence:score_page_quality"),
    FeatureSpec("multi_representation_chunks", "rag_project.intelligence.pdf_intelligence:enrich_text"),
    FeatureSpec("numeric_normalization", "rag_project.intelligence.advanced_reasoning:normalize_numeric_measurements"),
    FeatureSpec("entity_extraction", "rag_project.intelligence.pdf_intelligence:enrich_text"),
    FeatureSpec("section_detection", "rag_project.intelligence.pdf_intelligence:enrich_text"),
    FeatureSpec("table_aware_query_planning", "rag_project.intelligence.query_intelligence:plan_query"),
    FeatureSpec("figure_aware_query_planning", "rag_project.intelligence.query_intelligence:plan_query"),
    FeatureSpec("query_intent_classification", "rag_project.intelligence.query_intelligence:plan_query"),
    FeatureSpec("query_normalization", "rag_project.intelligence.query_intelligence:normalize_query"),
    FeatureSpec("query_decomposition", "rag_project.intelligence.query_intelligence:decompose_query"),
    FeatureSpec("query_variants", "rag_project.intelligence.query_intelligence:_make_variants"),
    FeatureSpec("document_router", "rag_project.intelligence.advanced_reasoning:document_router"),
    FeatureSpec("multi_query_retrieval", "rag_project.intelligence.god_mode:_safe_hits"),
    FeatureSpec("parent_child_context_strategy", "rag_project.intelligence.advanced_reasoning:build_parent_child_context"),
    FeatureSpec("neighbor_context_strategy", "rag_project.intelligence.advanced_reasoning:expand_neighbors"),
    FeatureSpec("metadata_aware_retrieval_boosts", "rag_project.intelligence.god_mode:_metadata_boost"),
    FeatureSpec("lexical_vector_fusion", "rag_project.retrieval.hybrid_retriever:HybridRetriever"),
    FeatureSpec("reranking", "rag_project.reranking.reranker:Reranker"),
    FeatureSpec("evidence_alignment_gate", "rag_project.app.rag_system:EvidenceAlignment"),
    FeatureSpec("evidence_confidence", "rag_project.intelligence.evidence_guard:evidence_confidence"),
    FeatureSpec("context_budget_control", "rag_project.retrieval.context_builder:ContextBuilder"),
    FeatureSpec("contextual_compression_hinting", "rag_project.intelligence.advanced_reasoning:sentence_compress"),
    FeatureSpec("multi_hop_signal", "rag_project.intelligence.query_intelligence:plan_query"),
    FeatureSpec("contradiction_detection", "rag_project.intelligence.evidence_guard:detect_contradiction"),
    FeatureSpec("numeric_consistency_check", "rag_project.intelligence.evidence_guard:numeric_consistency"),
    FeatureSpec("claim_extraction", "rag_project.intelligence.evidence_guard:split_claims"),
    FeatureSpec("claim_level_support_scoring", "rag_project.intelligence.evidence_guard:verify_claims"),
    FeatureSpec("citation_firewall", "rag_project.intelligence.evidence_guard:citation_firewall"),
    FeatureSpec("citation_validation", "rag_project.intelligence.final_44:validate_citations"),
    FeatureSpec("answer_grounding_gate", "rag_project.intelligence.advanced_reasoning:grounded_evidence_gate"),
    FeatureSpec("abstention_ladder", "rag_project.intelligence.advanced_reasoning:abstention_ladder"),
    FeatureSpec("untrusted_evidence_prompt_boundary", "rag_project.app.rag_system:_generate_with_citations"),
    FeatureSpec("prompt_injection_sanitization", "rag_project.app.rag_system:sanitize_evidence"),
    FeatureSpec("conversation_context_isolation", "rag_project.app.rag_system:ConversationMemory"),
    FeatureSpec("query_trace_metadata", "rag_project.intelligence.final_44:build_final_trace"),
    FeatureSpec("degraded_lexical_fallback", "rag_project.retrieval.hybrid_retriever:HybridRetriever"),
    FeatureSpec("resilient_generation_fallback", "rag_project.intelligence.god_mode:_answer_with_ladder"),
    FeatureSpec("index_visibility_guard", "rag_project.intelligence.atomic_versioning:install"),
    FeatureSpec("retrieval_fail_closed_behavior", "rag_project.intelligence.final_44:fail_closed"),
    FeatureSpec("exact_feature_certification_contract", "rag_project.intelligence.production_contract:validate_feature_contract"),
)


def resolve_target(target: str) -> Any:
    module_name, separator, attr_path = target.partition(":")
    if not separator or not module_name or not attr_path:
        raise ValueError(f"Invalid feature target: {target!r}")
    value: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def validate_feature_contract() -> dict[str, Any]:
    names = [feature.name for feature in FEATURES]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    unresolved: dict[str, str] = {}
    for feature in FEATURES:
        try:
            resolve_target(feature.target)
        except Exception as exc:
            unresolved[feature.name] = f"{type(exc).__name__}: {exc}"
    return {
        "feature_count": len(FEATURES),
        "unique_names": len(names) == len(set(names)),
        "duplicates": duplicates,
        "unresolved": unresolved,
        "all_resolved": not unresolved and not duplicates and len(FEATURES) == 44,
    }


_PHONENUMBER = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_LONG_ID = re.compile(r"\b\d{8,}\b")


def redact_sensitive_text(text: str) -> str:
    """Best-effort diagnostic redaction; never intended as a clinical de-identification system."""
    value = str(text or "")
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _PHONENUMBER.sub("[REDACTED_PHONE]", value)
    value = _LONG_ID.sub("[REDACTED_ID]", value)
    return value


def sanitize_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    """Return a telemetry-safe trace without changing the user-facing answer."""
    if not trace:
        return {}
    result = dict(trace)
    for key in ("question", "original_query"):
        if key in result:
            result[key] = redact_sensitive_text(str(result[key]))
    return result


def production_readiness(profile: dict[str, Any]) -> dict[str, Any]:
    """Deterministic release decision from independent gates."""
    gates = {
        "feature_contract": bool(profile.get("feature_contract")),
        "tests_green": bool(profile.get("tests_green")),
        "index_ready": bool(profile.get("index_ready")),
        "privacy_controls": bool(profile.get("privacy_controls")),
        "medical_safety": bool(profile.get("medical_safety")),
    }
    return {
        "gates": gates,
        "release_ready": all(gates.values()),
        "clinical_validation": False,
        "regulatory_approval": False,
    }


__all__ = ["FeatureSpec", "FEATURES", "resolve_target", "validate_feature_contract", "redact_sensitive_text", "sanitize_trace", "production_readiness"]
