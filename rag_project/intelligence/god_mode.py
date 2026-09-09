from __future__ import annotations

import re
import time
import uuid
from typing import Any, Sequence

from rag_project.intelligence.atomic_versioning import install as install_atomic_versioning
from rag_project.intelligence.evidence_guard import citation_firewall, contradiction_report, evidence_confidence, grounding_decision, verify_claims
from rag_project.intelligence.index_auditor import audit_index
from rag_project.intelligence.pdf_intelligence import enrich_text
from rag_project.intelligence.query_intelligence import QueryPlan, plan_query

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
    "resilient_generation_fallback", "index_visibility_guard", "retrieval_fail_closed_behavior", "exact_feature_certification_contract",
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:1600]


def _metadata_boost(hit: Any, plan: QueryPlan) -> float:
    meta = hit.metadata or {}
    types = {str(x).casefold() for x in (meta.get("evidence_types") or [])}
    boost = 1.0
    if plan.needs_table and ("table" in types or meta.get("table_id") or meta.get("is_table")):
        boost *= plan.score_boosts.get("table", 1.0)
    if plan.needs_figure and ("figure" in types or meta.get("figure_id") or meta.get("image_id")):
        boost *= plan.score_boosts.get("figure", 1.0)
    if plan.needs_multi_hop and (meta.get("parent_id") or meta.get("section_id") or meta.get("section_title")):
        boost *= plan.score_boosts.get("parent", 1.0)
    enriched = enrich_text(str(hit.text or ""))
    if set(plan.entities).intersection(enriched.get("keywords") or []):
        boost *= plan.score_boosts.get("entity", 1.0)
    return boost


def _diversify(hits: Sequence[Any], limit: int) -> list[Any]:
    chosen: list[Any] = []
    docs: dict[str, int] = {}
    pages: set[tuple[str, str]] = set()
    for hit in sorted(hits, key=lambda h: float(getattr(h, "score", 0.0)), reverse=True):
        meta = hit.metadata or {}
        doc = str(meta.get("document_id") or getattr(hit, "doc_id", ""))
        page_values = meta.get("page_numbers") or []
        page = str(page_values[0]) if page_values else "?"
        if docs.get(doc, 0) >= 3 or ((doc, page) in pages and len(chosen) < limit // 2):
            continue
        docs[doc] = docs.get(doc, 0) + 1
        pages.add((doc, page))
        chosen.append(hit)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for hit in sorted(hits, key=lambda h: float(getattr(h, "score", 0.0)), reverse=True):
            if hit not in chosen:
                chosen.append(hit)
            if len(chosen) >= limit:
                break
    return chosen


def _safe_hits(system: Any, plan: QueryPlan, where: dict[str, Any] | None) -> list[Any]:
    all_hits: dict[str, Any] = {}
    variants = list(dict.fromkeys([plan.normalized, *plan.variants, *plan.subqueries]))
    for variant in variants[: max(2, int(getattr(system.settings, "max_query_variants", 8)))]:
        try:
            hits = system.retriever.retrieve(_clean(variant), top_k=max(8, int(system.settings.top_k) * int(getattr(system.settings, "retrieval_candidate_multiplier", 5))), where=where)
        except Exception as exc:
            system.logger.warning("Retrieval branch failed: %s", exc)
            continue
        for hit in hits:
            key = str(hit.metadata.get("chunk_id") or hit.doc_id or str(hit.text)[:80])
            if key not in all_hits or float(hit.score) > float(all_hits[key].score):
                all_hits[key] = hit
    candidates = list(all_hits.values())
    if not candidates:
        return []
    try:
        candidates = system.reranker.rerank(plan.normalized, candidates)
    except Exception as exc:
        system.logger.warning("Reranking failed; retaining fused candidates: %s", exc)
    for hit in candidates:
        hit.score = float(hit.score) * _metadata_boost(hit, plan)
    return _diversify(candidates, max(int(system.settings.top_k) * 3, 12))


def _sanitize_hits(hits: Sequence[Any]) -> list[Any]:
    from rag_project.app.rag_system import RetrievalHit, sanitize_evidence
    result: list[Any] = []
    for hit in hits:
        text = sanitize_evidence(str(hit.text or ""))
        if text == hit.text:
            result.append(hit)
        else:
            meta = dict(hit.metadata or {})
            meta["evidence_sanitized"] = True
            result.append(RetrievalHit(hit.doc_id, text, meta, hit.score, hit.vector_score, hit.lexical_score))
    return result


def _answer_with_ladder(system: Any, question: str, context: str, selected_hits: Sequence[Any], conversation_context: str) -> tuple[str, str]:
    from rag_project.app.rag_system import _generate_with_citations
    try:
        answer, _ = _generate_with_citations(system.llm, question=question, context=context, selected_hits=selected_hits, conversation_context=conversation_context, temperature=system.settings.temperature)
        return answer, "primary"
    except Exception as exc:
        system.logger.warning("Primary generation failed: %s", exc)
    try:
        answer = system.llm.generate(prompt="Return only directly supported facts from the evidence. Use [S#] markers.\n\n" + context, system_prompt="You are an extractive evidence verifier. Never invent facts.", temperature=0.0)
        return answer, "extractive_fallback"
    except Exception as exc:
        system.logger.warning("Fallback generation failed: %s", exc)
        return "The language model is unavailable; I cannot safely generate a grounded answer right now.", "abstained"


def _god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    plan = plan_query(question)
    if not plan.normalized:
        return {"status": "LOW_QUALITY_QUERY", "answer": "Please provide a precise question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict()}
    try:
        self._ensure_embedding_dimension()
        compatibility = self.vector_store.compatibility_report(self.embedding_service.identity)
    except Exception as exc:
        return {"status": "INTERNAL_ERROR", "answer": "The retrieval index is temporarily unavailable.", "error": str(exc), "citations": [], "hits": [], "confidence": {"level": "unavailable", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict()}
    if str(compatibility.get("status", "")).upper() != "READY":
        return {"status": compatibility.get("status", "NOT_READY"), "answer": compatibility.get("message", "The index is not ready."), "citations": [], "hits": [], "confidence": {"level": "unavailable", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict()}
    try:
        from rag_project.retrieval.metadata_filter import MetadataFilter
        where = MetadataFilter.build(metadata_filter)
    except Exception:
        where = None
    hits = _sanitize_hits(_safe_hits(self, plan, where))
    if not hits:
        return {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient evidence in the indexed documents to answer this question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict()}
    try:
        alignment = self.evaluate_evidence_alignment(plan.normalized, hits[: max(int(self.settings.top_k) * 3, 12)])
    except Exception:
        alignment = {"decision": "PARTIALLY_SUPPORTED", "answerability": 0.25, "local_context_strength": 0.2, "contradiction": 0.0}
    if alignment.get("decision") == "NOT_SUPPORTED":
        return {"status": "NOT_SUPPORTED", "answer": "The indexed evidence does not directly support this question.", "citations": [], "hits": hits[:self.settings.top_k], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    selected = hits[: max(int(self.settings.top_k) * 3, 12)]
    try:
        context, selected = self.context_builder.build(selected)
    except Exception:
        context = "\n\n".join(f"<evidence id=\"S{i + 1}\">{h.text}</evidence>" for i, h in enumerate(selected[:self.settings.top_k]))
    answer, generation_path = _answer_with_ladder(self, question, context, selected, self.conversation_memory.prompt_context())
    blocks = [str(h.text or "") for h in selected]
    source_ids = [f"S{i + 1}" for i in range(len(selected))]
    claims = verify_claims(answer, blocks, source_ids)
    ground = grounding_decision(claims, min_supported_ratio=0.60)
    safe_answer, firewall_used = citation_firewall(answer, claims)
    contradiction = contradiction_report(claims)
    evidence_conf = evidence_confidence(retrieval=min(1.0, max((float(h.score) for h in selected), default=0.0)), rerank=min(1.0, max((float(h.score) for h in selected), default=0.0)), entailment=sum(c.support for c in claims) / max(len(claims), 1) if claims else 0.0, quality=sum(min(1.0, len(enrich_text(h.text).get("keywords") or []) / 12.0) for h in selected) / max(len(selected), 1), contradiction=1.0 if contradiction.get("has_contradiction") else 0.0)
    if not ground.get("allow") or contradiction.get("has_contradiction"):
        safe_answer = "I could not verify a sufficiently grounded answer from the indexed evidence. Unsupported or conflicting details were withheld."
    citations = self.citation_manager.validate(self.citation_manager.build(selected), selected)
    query_id = str(uuid.uuid4())
    trace = {"query_id": query_id, "original_query": question, "rewritten_query": plan.normalized, "retrieval_method": "multi_query_hybrid", "candidate_count": len(hits), "selected_count": len(selected), "generation_path": generation_path, "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}, "claims": [c.to_dict() for c in claims], "grounding": ground, "contradiction_report": contradiction, "firewall_used": firewall_used}
    try:
        self.state_store.record_query_trace(query_id, trace)
    except Exception:
        self.logger.exception("Failed to record query trace")
    self.conversation_memory.add(question, safe_answer)
    return {"query_id": query_id, "status": "SUCCESS" if ground.get("allow") and generation_path != "abstained" and not contradiction.get("has_contradiction") else "SUCCESS_WITH_WARNINGS", "answer": safe_answer, "citations": citations, "hits": list(selected), "confidence": {"level": "high" if evidence_conf >= 0.75 else "medium" if evidence_conf >= 0.50 else "low", "evidence_confidence": evidence_conf}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment, "grounding": ground, "claims": [c.to_dict() for c in claims], "contradiction_report": contradiction, "god_mode": True}


def report() -> dict[str, object]:
    return {"name": "GOD_MODE_RAG", "feature_count": len(GOD_MODE_FEATURES), "features": list(GOD_MODE_FEATURES), "fail_closed": True, "universal_pdf_mode": True, "answer_monkey_patch": False, "composition": "ProductionRAGSystem"}


def audit_god_mode_index(system: Any) -> dict[str, Any]:
    return audit_index(system)


def install() -> None:
    """Install index-version infrastructure only; never patch RAGSystem.answer."""
    install_atomic_versioning()
