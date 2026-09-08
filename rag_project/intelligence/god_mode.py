from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any, Sequence

from rag_project.intelligence.atomic_versioning import install as install_atomic_versioning
from rag_project.intelligence.evidence_guard import (
    citation_firewall,
    contradiction_report,
    evidence_confidence,
    grounding_decision,
    verify_claims,
)
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
    "resilient_generation_fallback", "index_visibility_guard", "retrieval_fail_closed_behavior",
)

_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _clean_variant(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:1600]


def _metadata_boost(hit: Any, plan: QueryPlan) -> float:
    meta = hit.metadata or {}
    types = {str(x).casefold() for x in (meta.get("evidence_types") or [])}
    boost = 1.0
    if plan.needs_table and ("table" in types or meta.get("table_id") or meta.get("is_table")):
        boost *= plan.score_boosts.get("table", 1.0)
    if plan.needs_figure and ("figure" in types or meta.get("figure_id") or meta.get("image_id")):
        boost *= plan.score_boosts.get("figure", 1.0)
    if plan.needs_neighborhood and meta.get("page_numbers"):
        boost *= 1.02
    if plan.needs_multi_hop and (meta.get("parent_id") or meta.get("section_id") or meta.get("section_title")):
        boost *= plan.score_boosts.get("parent", 1.0)
    enriched = enrich_text(str(hit.text or ""))
    entity_set = set(plan.entities)
    if entity_set and entity_set.intersection(enriched.get("keywords") or []):
        boost *= plan.score_boosts.get("entity", 1.0)
    return boost


def _diversify(candidates: Sequence[Any], limit: int) -> list[Any]:
    chosen: list[Any] = []
    seen_docs: dict[str, int] = {}
    seen_pages: set[tuple[str, str]] = set()
    for hit in sorted(candidates, key=lambda h: h.score, reverse=True):
        meta = hit.metadata or {}
        doc = str(meta.get("document_id") or hit.doc_id)
        pages = meta.get("page_numbers") or []
        page = str(pages[0]) if pages else "?"
        if seen_docs.get(doc, 0) >= 3:
            continue
        # Prefer source/page diversity until the result pool is useful.
        if (doc, page) in seen_pages and len(chosen) < limit // 2:
            continue
        seen_docs[doc] = seen_docs.get(doc, 0) + 1
        seen_pages.add((doc, page))
        chosen.append(hit)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for hit in sorted(candidates, key=lambda h: h.score, reverse=True):
            if hit not in chosen:
                chosen.append(hit)
            if len(chosen) >= limit:
                break
    return chosen


def _safe_hits(system: Any, plan: QueryPlan, where: dict[str, Any] | None) -> list[Any]:
    all_hits: dict[str, Any] = {}
    variants = list(dict.fromkeys([plan.normalized, *plan.variants, *plan.subqueries]))
    cap = max(2, int(getattr(system.settings, "max_query_variants", 8)))
    for variant in variants[:cap]:
        try:
            hits = system.retriever.retrieve(
                _clean_variant(variant),
                top_k=max(8, system.settings.top_k * getattr(system.settings, "retrieval_candidate_multiplier", 5)),
                where=where,
            )
        except Exception as exc:
            system.logger.warning("God-mode retrieval branch failed: %s", exc)
            continue
        for hit in hits:
            key = str(hit.metadata.get("chunk_id") or hit.doc_id or hit.text[:80])
            if key not in all_hits or hit.score > all_hits[key].score:
                all_hits[key] = hit
    candidates = list(all_hits.values())
    if not candidates:
        return []
    try:
        reranked = system.reranker.rerank(plan.normalized, candidates)
    except Exception as exc:
        system.logger.warning("God-mode reranking failed; using fused candidates: %s", exc)
        reranked = candidates
    for hit in reranked:
        hit.score *= _metadata_boost(hit, plan)
    return _diversify(reranked, max(system.settings.top_k * 3, 12))


def _sanitize_hits(hits: Sequence[Any]) -> list[Any]:
    from rag_project.app.rag_system import RetrievalHit, sanitize_evidence
    out: list[Any] = []
    for hit in hits:
        text = sanitize_evidence(str(hit.text or ""))
        if text == hit.text:
            out.append(hit)
        else:
            meta = dict(hit.metadata or {})
            meta["evidence_sanitized"] = True
            out.append(RetrievalHit(hit.doc_id, text, meta, hit.score, hit.vector_score, hit.lexical_score))
    return out


def _compress_hit_text(text: str, query: str, max_chars: int = 2400) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    scored: list[tuple[float, str]] = []
    query_terms = set(re.findall(r"[\wÀ-ÿ]{3,}", query.casefold()))
    for sentence in sentences:
        terms = set(re.findall(r"[\wÀ-ÿ]{3,}", sentence.casefold()))
        overlap = len(query_terms & terms) / max(len(query_terms), 1)
        numeric = 0.12 if re.search(r"\d", sentence) and re.search(r"(?:mg|ml|%|bpm|mmhg|kg)\b", sentence, re.I) else 0.0
        scored.append((overlap + numeric, sentence.strip()))
    selected: list[str] = []
    used = 0
    for _, sentence in sorted(scored, reverse=True):
        if not sentence:
            continue
        if used + len(sentence) + 1 > max_chars:
            continue
        selected.append(sentence)
        used += len(sentence) + 1
    return " ".join(reversed(selected)) if selected else text[:max_chars]


def _build_compressed_context(system: Any, hits: Sequence[Any], plan: QueryPlan) -> tuple[str, list[Any]]:
    compressed: list[Any] = []
    for hit in hits:
        text = _compress_hit_text(hit.text, plan.normalized, max_chars=2400)
        if text != hit.text:
            from rag_project.app.rag_system import RetrievalHit
            meta = dict(hit.metadata or {})
            meta["context_compressed"] = True
            hit = RetrievalHit(hit.doc_id, text, meta, hit.score, hit.vector_score, hit.lexical_score)
        compressed.append(hit)
    context, selected = system.context_builder.build(compressed)
    return context, selected


def _answer_with_ladder(system: Any, question: str, context: str, selected_hits: Sequence[Any], conversation_context: str) -> tuple[str, str]:
    from rag_project.app.rag_system import _generate_with_citations
    try:
        answer, _ = _generate_with_citations(
            system.llm, question=question, context=context, selected_hits=selected_hits,
            conversation_context=conversation_context, temperature=system.settings.temperature,
        )
        return answer, "primary"
    except Exception as exc:
        system.logger.warning("Primary generation failed: %s", exc)
    # Resilient fallback: lower temperature and explicitly request extractive facts only.
    fallback_prompt = (
        "Return only directly supported facts from the evidence. Preserve numbers and negations. "
        "Use source markers [S#]. Do not add background knowledge.\n\n" + context
    )
    try:
        answer = system.llm.generate(
            prompt=fallback_prompt,
            system_prompt="You are an extractive evidence verifier. Never invent facts.",
            temperature=0.0,
        )
        return answer, "extractive_fallback"
    except Exception as exc:
        system.logger.warning("Fallback generation failed: %s", exc)
        return "The language model is unavailable; I cannot safely generate a grounded answer right now.", "abstained"


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
    hits = _safe_hits(self, plan, where)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    hits = _sanitize_hits(hits)
    if not hits:
        return {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient evidence in the indexed documents to answer this question.", "citations": [], "hits": [], "confidence": {"level": "none", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": {"decision": "NOT_SUPPORTED"}}
    try:
        alignment = self.evaluate_evidence_alignment(plan.normalized, hits[: max(self.settings.top_k * 3, 12)])
    except Exception:
        alignment = {"decision": "PARTIALLY_SUPPORTED", "answerability": 0.25, "local_context_strength": 0.2, "contradiction": 0.0}
    if alignment.get("decision") == "NOT_SUPPORTED":
        return {"status": "NOT_SUPPORTED", "answer": "The indexed evidence does not directly support this question.", "citations": [], "hits": hits[:self.settings.top_k], "confidence": {"level": "none", "top_score": hits[0].score, "margin": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    selected_pool = hits[: max(self.settings.top_k * 3, 12)]
    try:
        context, selected_hits = _build_compressed_context(self, selected_pool, plan)
    except Exception as exc:
        self.logger.warning("God-mode context build failed: %s", exc)
        selected_hits = selected_pool[: self.settings.top_k]
        context = "\n\n".join(f"<evidence id=\"S{i + 1}\">{h.text}</evidence>" for i, h in enumerate(selected_hits))
    if not selected_hits:
        return {"status": "NOT_SUPPORTED", "answer": "The available evidence is insufficient to answer safely.", "citations": [], "hits": [], "confidence": {"level": "none", "top_score": 0.0, "margin": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    top = float(selected_hits[0].score)
    second = float(selected_hits[1].score) if len(selected_hits) > 1 else 0.0
    margin = max(0.0, top - second)
    quality = sum(min(1.0, len(enrich_text(h.text).get("keywords") or []) / 12.0) for h in selected_hits) / max(len(selected_hits), 1)
    pre_conf = evidence_confidence(
        retrieval=min(1.0, top), rerank=min(1.0, max(h.score for h in selected_hits)),
        entailment=max(float(alignment.get("answerability", 0.0)), float(alignment.get("local_context_strength", 0.0))),
        quality=quality, contradiction=float(alignment.get("contradiction", 0.0)),
    )
    if pre_conf < float(getattr(self.settings, "evidence_min_confidence", 0.25)):
        return {"status": "NOT_SUPPORTED", "answer": "I do not have enough directly relevant evidence in the indexed documents to answer this reliably.", "citations": [], "hits": list(selected_hits), "confidence": {"level": "low", "top_score": top, "margin": margin, "evidence_confidence": pre_conf}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment}
    conversation_context = self.conversation_memory.prompt_context()
    generation_started = time.perf_counter()
    answer, generation_path = _answer_with_ladder(self, question, context, selected_hits, conversation_context)
    generation_ms = (time.perf_counter() - generation_started) * 1000
    evidence_blocks = [str(h.text or "") for h in selected_hits]
    source_ids = [f"S{i + 1}" for i in range(len(selected_hits))]
    claims = verify_claims(answer, evidence_blocks, source_ids)
    ground = grounding_decision(claims, min_supported_ratio=0.60)
    safe_answer, firewall_used = citation_firewall(answer, claims)
    contradiction = contradiction_report(claims)
    unsafe = [c for c in claims if c.status in {"UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"}]
    supported = [c for c in claims if c.status in {"SUPPORTED", "PARTIAL"}]
    entailment = sum(c.support for c in claims) / max(len(claims), 1) if claims else 0.0
    final_conf = evidence_confidence(
        retrieval=min(1.0, top), rerank=min(1.0, max(h.score for h in selected_hits)), entailment=entailment,
        quality=quality, contradiction=1.0 if contradiction["has_contradiction"] else 0.0,
    )
    if not ground["allow"]:
        safe_answer = "I could not verify a sufficiently grounded answer from the indexed evidence. The generated details were withheld rather than presented as facts."
    self.conversation_memory.add(question, safe_answer)
    citations = self.citation_manager.validate(self.citation_manager.build(selected_hits), selected_hits)
    query_id = str(uuid.uuid4())
    trace = {
        "original_query": question, "rewritten_query": plan.normalized, "retrieval_method": "god_mode_multi_query_hybrid",
        "candidate_count": len(hits), "selected_count": len(selected_hits), "generation_path": generation_path,
        "timings_ms": {"retrieval": round(retrieval_ms, 3), "generation": round(generation_ms, 3), "total": round((time.perf_counter() - started) * 1000, 3)},
        "query_plan": plan.to_dict(), "claims": [c.to_dict() for c in claims], "grounding": ground,
        "contradiction_report": contradiction, "confidence": {"evidence": final_conf, "top_score": top, "margin": margin},
        "firewall_used": firewall_used, "selected_chunk_ids": [h.metadata.get("chunk_id", h.doc_id) for h in selected_hits],
    }
    try:
        self.state_store.record_query_trace(query_id, trace)
    except Exception:
        self.logger.exception("Failed to record god-mode query trace")
    return {
        "query_id": query_id,
        "status": "SUCCESS" if ground["allow"] and generation_path != "abstained" else "SUCCESS_WITH_WARNINGS",
        "answer": safe_answer,
        "citations": citations,
        "hits": list(selected_hits),
        "confidence": {"level": "high" if final_conf >= 0.75 else "medium" if final_conf >= 0.50 else "low", "evidence_confidence": final_conf, "top_score": top, "margin": margin, "claims_supported": len(supported), "claims_unsafe": len(unsafe)},
        "query_analysis": plan.to_dict(), "evidence_alignment": alignment, "grounding": ground,
        "claims": [c.to_dict() for c in claims], "contradiction_report": contradiction, "god_mode": True,
    }


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
        if not hasattr(RAGSystem, "god_mode_report"):
            RAGSystem.god_mode_report = staticmethod(report)
        if not hasattr(RAGSystem, "audit_god_mode_index"):
            RAGSystem.audit_god_mode_index = audit_index
        _INSTALLED = True
