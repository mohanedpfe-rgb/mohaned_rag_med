from __future__ import annotations

import re
import threading
import time
from typing import Any, Sequence

from rag_project.intelligence.evidence_guard import evidence_confidence, citation_firewall, verify_claims
from rag_project.intelligence.query_intelligence import QueryPlan, plan_query
from rag_project.intelligence.pdf_intelligence import enrich_text
from rag_project.intelligence.atomic_versioning import install as install_atomic_versioning


GOD_MODE_FEATURES = (
    "universal_pdf_routing",
    "page_quality_scoring",
    "adaptive_ocr_routing",
    "alternate_extractor_trigger",
    "ocr_quality_detection",
    "multi_representation_chunks",
    "numeric_normalization",
    "entity_extraction",
    "section_detection",
    "table_aware_query_planning",
    "figure_aware_query_planning",
    "query_intent_classification",
    "query_normalization",
    "query_decomposition",
    "query_variants",
    "document_router",
    "multi_query_retrieval",
    "parent_child_context_strategy",
    "neighbor_context_strategy",
    "metadata_aware_retrieval_boosts",
    "lexical_vector_fusion",
    "reranking",
    "evidence_alignment_gate",
    "evidence_confidence",
    "context_budget_control",
    "contextual_compression_hinting",
    "multi_hop_signal",
    "contradiction_detection",
    "numeric_consistency_check",
    "claim_extraction",
    "claim_level_support_scoring",
    "citation_firewall",
    "citation_validation",
    "answer_grounding_gate",
    "abstention_ladder",
    "untrusted_evidence_prompt_boundary",
    "prompt_injection_sanitization",
    "conversation_context_isolation",
    "query_trace_metadata",
    "degraded_lexical_fallback",
    "resilient_generation_fallback",
    "index_visibility_guard",
    "retrieval_fail_closed_behavior",
)

_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _clean_variant(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value[:1600]


def _safe_hits(system: Any, query: str, plan: QueryPlan, where: dict[str, Any] | None) -> list[Any]:
    all_hits: dict[str, Any] = {}
    queries = list(plan.variants) + list(plan.subqueries)
    for variant in queries[: max(2, int(getattr(system.settings, "max_query_variants", 8)))]:
        try:
            hits = system.retriever.retrieve(_clean_variant(variant), top_k=max(8, system.settings.top_k * getattr(system.settings, "retrieval_candidate_multiplier", 5)), where=where)
        except Exception as exc:
            system.logger.warning("God-mode retrieval branch failed: %s", exc)
            continue
        for hit in hits:
            key = str(hit.metadata.get("chunk_id") or hit.doc_id or hit.text[:80])
            existing = all_hits.get(key)
            if existing is None or hit.score > existing.score:
                all_hits[key] = hit
    candidates = list(all_hits.values())
    if not candidates:
        return []
    try:
        reranked = system.reranker.rerank(plan.normalized, candidates)
    except Exception as exc:
        system.logger.warning("God-mode reranking failed; using fused candidates: %s", exc)
        reranked = candidates
    boosted = []
    for hit in reranked:
        meta = hit.metadata or {}
        boost = 1.0
        types = {str(x).lower() for x in (meta.get("evidence_types") or [])}
        if plan.needs_table and ("table" in types or meta.get("table_id")):
            boost *= 1.15
        if any(term in hit.text.casefold() for term in plan.entities):
            boost *= 1.06
        if plan.needs_neighborhood and meta.get("page_numbers"):
            boost *= 1.02
        hit.score *= boost
        boosted.append(hit)
    boosted.sort(key=lambda h: h.score, reverse=True)
    return boosted[: max(system.settings.top_k * 2, 8)]


def _sanitize_hits(hits: Sequence[Any]) -> list[Any]:
    from rag_project.app.rag_system import sanitize_evidence, RetrievalHit
    sanitized: list[Any] = []
    for hit in hits:
        text = sanitize_evidence(str(hit.text or ""))
        if text == hit.text:
            sanitized.append(hit)
        else:
            meta = dict(hit.metadata or {})
            meta["evidence_sanitized"] = True
            sanitized.append(RetrievalHit(doc_id=hit.doc_id, text=text, metadata=meta, score=hit.score, vector_score=hit.vector_score, lexical_score=hit.lexical_score))
    return sanitized


def _quality_from_hits(hits: Sequence[Any]) -> float:
    if not hits:
        return 0.0
    values = []
    for hit in hits:
        keywords = enrich_text(hit.text).get("keywords") or []
        values.append(min(1.0, len(keywords) / 12.0))
    return sum(values) / max(len(values), 1)


def _god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    plan = plan_query(question)
    if not plan.normalized:
        return {"status": "LOW_QUALITY_QUERY", "answer": "Please provide a precise question.", "citations": [], "hits": [], "confidence": {"level": "none", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict()}
    try:
        self._ensure_embedding_dimension()
        self.index_compatibility = self.vector_store.compatibility_report(self.embedding_service.identity)
    except Exception as exc:
        self.logger.exception("God-mode compatibility check failed")
        return {"status": "INTERNAL_ERROR", "answer": "The retrieval index is temporarily unavailable.", "error": str(exc), "citations": [], "hits": [], "confidence": {"level": "unavailable", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict()}
    if self.index_compatibility.get("status") != "READY":
        return {"status": self.index_compatibility.get("status", "NOT_READY"), "answer": self.index_compatibility.get("message", "The index is not ready."), "error": self.index_compatibility, "citations": [], "hits": [], "confidence": {"level": "unavailable", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict()}
    where = None
    try:
        from rag_project.retrieval.metadata_filter import MetadataFilter
        where = MetadataFilter.build(metadata_filter)
    except Exception:
        where = None

    retrieval_start = time.perf_counter()
    hits = _safe_hits(self, plan.normalized, plan, where)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    hits = _sanitize_hits(hits)
    if not hits:
        return {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient evidence in the indexed documents to answer this question.", "citations": [], "hits": [], "confidence": {"level": "none", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": {"decision": "NOT_SUPPORTED"}}
    try:
        alignment = self.evaluate_evidence_alignment(plan.normalized, hits[: max(self.settings.top_k * 2, 8)])
    except Exception:
        alignment = {"decision": "PARTIALLY_SUPPORTED", "answerability": 0.25, "local_context_strength": 0.2, "contradiction": 0.0}
    if alignment.get("decision") == "NOT_SUPPORTED":
        return {"status": "NOT_SUPPORTED", "answer": "The indexed evidence does not directly support this question. Please rephrase it or provide a more specific concept.", "citations": [], "hits": hits[:self.settings.top_k], "confidence": {"level": "none", "top_score": hits[0].score, "margin": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    selected = hits[: max(self.settings.top_k * 2, 8)]
    try:
        context, selected_hits = self.context_builder.build(selected)
    except Exception as exc:
        self.logger.warning("God-mode context build failed: %s", exc)
        selected_hits = selected[: self.settings.top_k]
        context = "\n\n".join(f"[S{i + 1}] {h.text}" for i, h in enumerate(selected_hits))
    if not selected_hits:
        return {"status": "NOT_SUPPORTED", "answer": "The available evidence is insufficient to answer safely.", "citations": [], "hits": [], "confidence": {"level": "none", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    top = float(selected_hits[0].score if selected_hits else 0.0)
    second = float(selected_hits[1].score if len(selected_hits) > 1 else 0.0)
    margin = max(0.0, top - second)
    answerable = float(alignment.get("answerability", 0.0))
    local = float(alignment.get("local_context_strength", 0.0))
    pre_conf = evidence_confidence(retrieval=min(1.0, top), rerank=min(1.0, max(h.score for h in selected_hits)), entailment=max(answerable, local), quality=_quality_from_hits(selected_hits), contradiction=float(alignment.get("contradiction", 0.0)))
    if pre_conf < float(getattr(self.settings, "evidence_min_confidence", 0.25)) or (answerable < 0.15 and local < 0.10):
        return {"status": "NOT_SUPPORTED", "answer": "I do not have enough directly relevant evidence in the indexed documents to answer this reliably.", "citations": [], "hits": list(selected_hits), "confidence": {"level": "low", "top_score": top, "margin": margin, "evidence_confidence": pre_conf}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    conversation_context = self.conversation_memory.prompt_context()
    generation_started = time.perf_counter()
    try:
        from rag_project.app.rag_system import _generate_with_citations
        answer, _ = _generate_with_citations(self.llm, question=question, context=context, selected_hits=selected_hits, conversation_context=conversation_context, temperature=self.settings.temperature)
    except Exception as exc:
        self.logger.warning("God-mode generation unavailable: %s", exc)
        answer = "The language model is currently unavailable. The verified evidence is listed in the citations."
    generation_ms = (time.perf_counter() - generation_started) * 1000
    evidence_blocks = [str(h.text or "") for h in selected_hits]
    source_ids = [f"S{i + 1}" for i in range(len(selected_hits))]
    claims = verify_claims(answer, evidence_blocks, source_ids)
    safe_answer, firewall_used = citation_firewall(answer, claims)
    unsafe = [c for c in claims if c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"}]
    supported = [c for c in claims if c.status == "SUPPORTED"]
    entailment = sum(c.support for c in claims) / max(len(claims), 1) if claims else 0.0
    contradiction = 1.0 if any(c.contradiction for c in claims) else 0.0
    final_conf = evidence_confidence(retrieval=min(1.0, top), rerank=min(1.0, max(h.score for h in selected_hits)), entailment=entailment, quality=_quality_from_hits(selected_hits), contradiction=contradiction)
    if unsafe and not supported:
        safe_answer = "I could not verify a safe, evidence-supported answer from the indexed documents. The relevant evidence is retained in the citations for inspection."
    self.conversation_memory.add(question, safe_answer)
    citations = self.citation_manager.validate(self.citation_manager.build(selected_hits), selected_hits)
    query_id = str(__import__('uuid').uuid4())
    try:
        self.state_store.record_query_trace(query_id, {"original_query": question, "rewritten_query": plan.normalized, "retrieval_method": "god_mode_multi_query_hybrid", "candidate_count": len(hits), "timings_ms": {"retrieval": round(retrieval_ms, 3), "generation": round(generation_ms, 3), "total": round((time.perf_counter() - started) * 1000, 3)}, "query_plan": plan.to_dict(), "claims": [c.to_dict() for c in claims], "confidence": {"evidence": final_conf, "top_score": top, "margin": margin}, "firewall_used": firewall_used, "selected_chunk_ids": [h.metadata.get("chunk_id", h.doc_id) for h in selected_hits]})
    except Exception:
        self.logger.exception("Failed to record god-mode query trace")
    return {"query_id": query_id, "status": "SUCCESS_WITH_WARNINGS" if unsafe or firewall_used else "SUCCESS", "answer": safe_answer, "citations": citations, "hits": list(selected_hits), "confidence": {"level": "high" if final_conf >= 0.75 else "medium" if final_conf >= 0.50 else "low", "evidence_confidence": final_conf, "top_score": top, "margin": margin, "claims_supported": len(supported), "claims_unsafe": len(unsafe)}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment, "claims": [c.to_dict() for c in claims], "god_mode": True}


def report() -> dict[str, Any]:
    return {"name": "GOD_MODE_RAG", "feature_count": len(GOD_MODE_FEATURES), "features": list(GOD_MODE_FEATURES), "fail_closed": True, "universal_pdf_mode": True}


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        install_atomic_versioning()
        from rag_project.app.rag_system import RAGSystem
        if not hasattr(RAGSystem, "_original_god_mode_answer"):
            RAGSystem._original_god_mode_answer = RAGSystem.answer
            RAGSystem.answer = _god_answer
        _INSTALLED = True
