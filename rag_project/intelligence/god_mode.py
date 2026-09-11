from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Sequence

from rag_project.intelligence.atomic_versioning import install as install_atomic_versioning
from rag_project.intelligence.advanced_clinical_reasoner import assess_clinical_reasoning, build_reasoning_instruction
from rag_project.intelligence.answer_repair import repair_and_verify
from rag_project.intelligence.evidence_guard import citation_firewall, contradiction_report, evidence_confidence, grounding_decision, verify_claims
from rag_project.intelligence.index_auditor import audit_index
from rag_project.intelligence.pdf_intelligence import enrich_text
from rag_project.intelligence.production_contract import sanitize_trace
from rag_project.intelligence.query_intelligence import QueryPlan, plan_query
from rag_project.retrieval.metadata_filter import MetadataFilter
from rag_project.intelligence.semantic_reasoning import build_evidence_graph, clinical_reasoning_ready, semantic_evidence_alignment, understand_query
from rag_project.intelligence.small_model_reasoner import analyze_with_small_model, augment_query_plan, merge_understanding

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

EXTENDED_GOD_MODE_CAPABILITIES = (
    "structured_semantic_understanding", "multilingual_entity_normalization", "contextual_follow_up_understanding",
    "typed_clinical_fact_extraction", "evidence_graph_reasoning", "explicit_path_search", "three_hop_reasoning",
    "conflict_aware_reasoning", "safety_aware_reasoning_gate", "calibrated_reasoning_confidence", "reasoning_contract",
    "small_model_semantic_copilot", "hard_query_escalation", "structured_query_hypothesis", "retrieval_expansion_without_authority",
    "evidence_claim_matrix", "critic_repair_loop", "post_repair_reverification", "adaptive_query_budgeting", "benchmark_harness", "confidence_calibration_contract",
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:1600]


def _logger(system: Any) -> logging.Logger:
    logger = getattr(system, "logger", None)
    return logger if logger is not None and hasattr(logger, "warning") else logging.getLogger(__name__)


def _metadata_boost(hit: Any, plan: QueryPlan) -> float:
    meta = hit.metadata or {}
    types = {str(x).casefold() for x in _as_list(meta.get("evidence_types"))}
    boost = 1.0
    if plan.needs_table and ("table" in types or meta.get("table_id") or meta.get("is_table")):
        boost *= plan.score_boosts.get("table", 1.0)
    if plan.needs_figure and ("figure" in types or meta.get("figure_id") or meta.get("image_id")):
        boost *= plan.score_boosts.get("figure", 1.0)
    if plan.needs_multi_hop and (meta.get("parent_id") or meta.get("section_id") or meta.get("section_title")):
        boost *= plan.score_boosts.get("parent", 1.0)
    enriched = enrich_text(str(hit.text or ""))
    if set(plan.entities).intersection(_as_list(enriched.get("keywords"))):
        boost *= plan.score_boosts.get("entity", 1.0)
    if plan.score_boosts.get("semantic", 1.0) > 1.0 and meta.get("language"):
        boost *= 1.02
    return boost


def _diversify(hits: Sequence[Any], limit: int) -> list[Any]:
    chosen: list[Any] = []
    docs: dict[str, int] = {}
    pages: set[tuple[str, str]] = set()
    for hit in sorted(hits, key=lambda h: float(getattr(h, "score", 0.0)), reverse=True):
        meta = hit.metadata or {}
        doc = str(meta.get("document_id") or getattr(hit, "doc_id", ""))
        page_values = _as_list(meta.get("page_numbers"))
        page = str(page_values[0]) if page_values else "?"
        if docs.get(doc, 0) >= 3 or ((doc, page) in pages and len(chosen) < limit // 2):
            continue
        docs[doc] = docs.get(doc, 0) + 1
        pages.add((doc, page))
        chosen.append(hit)
        if len(chosen) >= limit:
            break
    return chosen


def _bridge_variants(plan: QueryPlan) -> tuple[str, ...]:
    if not plan.needs_multi_hop or len(plan.entities) < 2:
        return ()
    anchors = [str(entity) for entity in plan.entities[:6] if entity]
    relations = {
        "causes": ("cause", "mechanism", "pathway"), "diagnosis": ("diagnosis", "criteria", "findings"),
        "management": ("treatment", "management", "indication"), "etiology": ("cause", "risk factor", "etiology"),
        "mechanism": ("mechanism", "pathway", "physiopathology"), "association": ("association", "relationship", "linked"),
        "relationship": ("association", "relationship", "linked"),
    }
    relation_terms = relations.get(plan.intent, ("association", "mechanism"))
    variants: list[str] = []
    for left, right in zip(anchors, anchors[1:]):
        for relation in relation_terms[:2]:
            variants.append(f"{left} {relation} {right}")
    return tuple(dict.fromkeys(variants))[:6]


def _safe_hits(system: Any, plan: QueryPlan, where: dict[str, Any] | None) -> list[Any]:
    all_hits: dict[str, Any] = {}
    variants = list(dict.fromkeys([plan.normalized, *plan.variants, *plan.subqueries]))
    max_variants = max(2, int(getattr(system.settings, "max_query_variants", 8)))
    variants = variants[:max_variants]
    if not variants:
        return []
    candidate_count = max(8, int(system.settings.top_k) * int(getattr(system.settings, "retrieval_candidate_multiplier", 5)))
    rerank_budget = 48
    bridge_variants = list(_bridge_variants(plan))
    initial_budget = max(1, rerank_budget - min(12, len(bridge_variants) * 2))
    base_share, remainder = divmod(initial_budget, len(variants))
    logger = _logger(system)
    for index, variant in enumerate(variants):
        variant_budget = min(candidate_count, max(1, base_share + (1 if index < remainder else 0)))
        try:
            hits = _as_list(system.retriever.retrieve(_clean(variant), top_k=variant_budget, where=where))
        except Exception as exc:
            logger.warning("Retrieval branch failed: %s", exc)
            continue
        for hit in hits:
            if hit is None:
                continue
            meta = hit.metadata or {}; key = str(meta.get("chunk_id") or hit.doc_id or str(hit.text)[:80])
            if key not in all_hits or float(hit.score) > float(all_hits[key].score):
                all_hits[key] = hit
    if bridge_variants:
        bridge_budget = max(1, rerank_budget - initial_budget)
        share, remainder = divmod(bridge_budget, len(bridge_variants))
        for index, variant in enumerate(bridge_variants):
            budget = max(1, share + (1 if index < remainder else 0))
            try:
                hits = _as_list(system.retriever.retrieve(_clean(variant), top_k=min(candidate_count, budget), where=where))
            except Exception as exc:
                logger.warning("Second-hop retrieval branch failed: %s", exc)
                continue
            for hit in hits:
                if hit is None:
                    continue
                meta = hit.metadata or {}; key = str(meta.get("chunk_id") or hit.doc_id or str(hit.text)[:80])
                if key not in all_hits or float(hit.score) > float(all_hits[key].score):
                    all_hits[key] = hit
    candidates = sorted(all_hits.values(), key=lambda hit: float(getattr(hit, "score", 0.0)), reverse=True)[:rerank_budget]
    if not candidates:
        return []
    base_scores = {str((hit.metadata or {}).get("chunk_id") or hit.doc_id or str(hit.text)[:80]): max(0.0, min(1.0, float(getattr(hit, "score", 0.0)))) for hit in candidates}
    try:
        reranked = _as_list(system.reranker.rerank(plan.normalized, candidates))
        if not reranked:
            raise ValueError("reranker_returned_no_candidates")
        for hit in reranked:
            key = str((hit.metadata or {}).get("chunk_id") or hit.doc_id or str(hit.text)[:80])
            rerank_score = max(0.0, min(1.0, float(getattr(hit, "score", 0.0))))
            hit.score = 0.60 * base_scores.get(key, rerank_score) + 0.40 * rerank_score
        candidates = reranked
    except Exception as exc:
        logger.warning("Reranking failed; retaining fused candidates: %s", exc)
        for hit in candidates:
            hit.score = max(0.0, min(1.0, float(getattr(hit, "score", 0.0))))
    for hit in candidates:
        hit.score = min(1.0, float(hit.score) * _metadata_boost(hit, plan))
    return _diversify(candidates, max(int(system.settings.top_k) * 3, 12))


def _sanitize_hits(hits: Sequence[Any]) -> list[Any]:
    from rag_project.app.rag_system import RetrievalHit, sanitize_evidence
    result: list[Any] = []
    for hit in _as_list(hits):
        if hit is None:
            continue
        text = sanitize_evidence(str(hit.text or ""))
        if text == hit.text:
            result.append(hit)
        else:
            meta = dict(hit.metadata or {}); meta["evidence_sanitized"] = True
            result.append(RetrievalHit(hit.doc_id, text, meta, hit.score, hit.vector_score, hit.lexical_score))
    return result


def _simple_extractive_answer(question: str, selected_hits: Sequence[Any], max_sentences: int = 6) -> str:
    question_terms = set(re.findall(r"[\wÀ-ÿ-]{3,}", str(question or "").casefold()))
    stop = {"what", "are", "the", "main", "findings", "is", "this", "that", "what", "does", "document", "report", "explain", "define", "list", "show", "about", "principal", "biais"}
    question_terms -= stop
    candidates: list[tuple[float, str]] = []
    for index, hit in enumerate(_as_list(selected_hits)):
        if hit is None:
            continue
        marker = f"[S{index + 1}]"
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", str(getattr(hit, "text", "") or "")):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence:
                continue
            metadata_match = re.match(r"^\[Section:[^\]]*\]\s*", sentence, re.I)
            cleaned = metadata_match.sub("", sentence).strip() if metadata_match else sentence
            if not cleaned or len(cleaned) < 12:
                continue
            tokens = set(re.findall(r"[\wÀ-ÿ-]{3,}", cleaned.casefold()))
            overlap = len(tokens & question_terms) / max(1, len(question_terms)) if question_terms else 0.0
            score = 0.60 * float(getattr(hit, "score", 0.0) or 0.0) + 0.40 * overlap
            candidates.append((score, f"{cleaned} {marker}"))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen: list[str] = []
    seen: set[str] = set()
    for score, sentence in candidates:
        normalized = re.sub(r"\[S\d+\]", "", sentence).casefold()
        if normalized in seen or score < 0.18:
            continue
        seen.add(normalized)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break
    return "\n".join(f"- {sentence}" for sentence in chosen)


def _answer_with_ladder(system: Any, question: str, context: str, selected_hits: Sequence[Any], conversation_context: str, reasoning_instruction: str = "") -> tuple[str, str]:
    from rag_project.app.rag_system import _generate_with_citations
    prompt_context = context + ("\n\n<reasoning_task>" + reasoning_instruction + "</reasoning_task>" if reasoning_instruction else "")
    hard_markers = ("why", "how does", "how do", "cause", "causes", "mechanism", "compare", "versus", "difference", "diagnostic criteria", "dose", "dosage", "treatment", "prognosis", "contraindication", "pourquoi", "comment", "سبب", "مقارنة", "علاج", "تشخيص", "جرعة")
    simple_path = not reasoning_instruction.strip() and not any(marker in str(question or "").casefold() for marker in hard_markers)
    if simple_path:
        extractive = _simple_extractive_answer(question, selected_hits)
        if extractive:
            return extractive, "fast_extractive"
    try:
        answer, _ = _generate_with_citations(system.llm, question=question, context=prompt_context, selected_hits=selected_hits, conversation_context=conversation_context, temperature=system.settings.temperature)
        return answer, "primary"
    except Exception as exc:
        _logger(system).warning("Primary generation failed: %s", exc)
    try:
        answer = system.llm.generate(prompt="Return only directly supported facts from the evidence. Use [S#] markers.\n\n" + prompt_context, system_prompt="You are an extractive evidence verifier. Never invent facts or clinical recommendations. For causal or multi-hop questions, connect only relationships explicitly supported by the evidence.", temperature=0.0)
        return answer, "extractive_fallback"
    except Exception as exc:
        _logger(system).warning("Fallback generation failed: %s", exc)
        return "The language model is unavailable; I cannot safely generate a grounded answer right now.", "abstained"


def _god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    conversation_context = self.conversation_memory.prompt_context()
    plan = plan_query(question, conversation_context=conversation_context)
    understanding = understand_query(question, conversation_context=conversation_context)
    small_model_assist = analyze_with_small_model(self.llm, question, understanding, conversation_context)
    plan = augment_query_plan(plan, small_model_assist)
    understanding = merge_understanding(understanding, small_model_assist)
    if not plan.normalized:
        return {"status": "LOW_QUALITY_QUERY", "answer": "Please provide a precise question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict()}
    try:
        self._ensure_embedding_dimension(); compatibility = self.vector_store.compatibility_report(self.embedding_service.identity)
    except Exception as exc:
        return {"status": "INTERNAL_ERROR", "answer": "The retrieval index is temporarily unavailable.", "error": type(exc).__name__, "citations": [], "hits": [], "confidence": {"level": "unavailable", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict()}
    if str(compatibility.get("status", "")).upper() != "READY":
        return {"status": compatibility.get("status", "NOT_READY"), "answer": compatibility.get("message", "The index is not ready."), "citations": [], "hits": [], "confidence": {"level": "unavailable", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict()}
    try:
        where = MetadataFilter.build(metadata_filter)
    except Exception:
        return {"status": "INVALID_FILTER", "answer": "The requested document filter is invalid.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict()}
    hits = _sanitize_hits(_safe_hits(self, plan, where))
    if not hits:
        return {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient evidence in the indexed documents to answer this question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict(), "small_model_assist": small_model_assist or {}}
    semantic_alignment = semantic_evidence_alignment(plan.normalized, hits, conversation_context=conversation_context)
    try:
        alignment = self.evaluate_evidence_alignment(plan.normalized, hits[: max(int(self.settings.top_k) * 3, 12)])
    except Exception as exc:
        self.logger.exception("Evidence-alignment evaluation failed"); alignment = {"decision": "NOT_SUPPORTED", "answerability": 0.0, "local_context_strength": 0.0, "contradiction": 0.0, "error": type(exc).__name__}
    combined_score = max(float(alignment.get("answerability", 0.0) or 0.0), float(semantic_alignment.get("score", 0.0) or 0.0))
    generic_factual = plan.intent in {"factual"} and not plan.entities and not understanding.relations
    if alignment.get("decision") in {"NOT_SUPPORTED", "RELATED_BUT_NOT_ANSWERING"} and combined_score < 0.25 and not generic_factual:
        return {"status": "NOT_SUPPORTED", "answer": "The indexed evidence does not directly support this question.", "citations": [], "hits": hits[:self.settings.top_k], "confidence": {"level": "none", "evidence_confidence": 0.0}, "query_analysis": plan.to_dict(), "evidence_alignment": alignment, "semantic_alignment": semantic_alignment, "semantic_understanding": understanding.to_dict(), "small_model_assist": small_model_assist or {}}
    selected = hits[: max(int(self.settings.top_k) * 3, 12)]
    try:
        context, selected = self.context_builder.build(selected)
    except Exception:
        context = "\n\n".join(f"<evidence id=\"S{i + 1}\">{h.text}</evidence>" for i, h in enumerate(selected[:self.settings.top_k]))
    nodes, edges = build_evidence_graph(selected, understanding)
    reasoning_state = clinical_reasoning_ready(understanding, nodes, edges)
    advanced_reasoning = assess_clinical_reasoning(understanding, selected, max_depth=3)
    reasoning_gate_required = bool(understanding.entities) or bool(understanding.relations) or understanding.primary_intent in {"comparison", "etiology", "mechanism", "diagnosis", "management", "prognosis", "relationship"}
    if not advanced_reasoning.allow_generation and reasoning_gate_required:
        return {"status": "REASONING_ABSTAIN", "answer": "The retrieved evidence is insufficient or conflicting for a safe clinical reasoning answer.", "citations": [], "hits": list(selected), "confidence": {"level": "low", "evidence_confidence": advanced_reasoning.confidence}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict(), "evidence_alignment": alignment, "semantic_alignment": semantic_alignment, "clinical_reasoning": reasoning_state, "advanced_reasoning": advanced_reasoning.to_dict(), "small_model_assist": small_model_assist or {}}
    reasoning_instruction = build_reasoning_instruction(understanding, advanced_reasoning)
    if small_model_assist and small_model_assist.get("answer_strategy"):
        reasoning_instruction += " Retrieval strategy hypothesis from the small model (never evidence): " + small_model_assist["answer_strategy"]
    answer, generation_path = _answer_with_ladder(self, question, context, selected, conversation_context, reasoning_instruction)
    blocks = [str(h.text or "") for h in selected]; source_ids = [f"S{i + 1}" for i in range(len(selected))]
    claims = verify_claims(answer, blocks, source_ids); pre_repair_claims = list(claims)
    ground = grounding_decision(claims, min_supported_ratio=0.60)
    safe_answer, firewall_used = citation_firewall(answer, claims); contradiction = contradiction_report(claims)
    repair_used = False
    if not ground.get("allow") or contradiction.get("has_contradiction"):
        repaired_answer, repaired_claims, attempted = repair_and_verify(self.llm, question, answer, blocks, source_ids, claims)
        if attempted:
            repair_used = True
            answer = repaired_answer
            claims = repaired_claims
            ground = grounding_decision(claims, min_supported_ratio=0.60)
            safe_answer, firewall_used = citation_firewall(answer, claims)
            contradiction = contradiction_report(claims)
    evidence_conf = evidence_confidence(retrieval=min(1.0, max((float(h.score) for h in selected), default=0.0)), rerank=min(1.0, max((float(h.score) for h in selected), default=0.0)), entailment=sum(c.support for c in claims) / max(len(claims), 1) if claims else 0.0, quality=sum(min(1.0, len(_as_list(enrich_text(h.text).get("keywords"))) / 12.0) for h in selected) / max(len(selected), 1), contradiction=1.0 if contradiction.get("has_contradiction") else 0.0)
    if not ground.get("allow") or contradiction.get("has_contradiction"):
        safe_answer = "I could not verify a sufficiently grounded answer from the indexed evidence. Unsupported or conflicting details were withheld."
    citations = self.citation_manager.validate(self.citation_manager.build(selected), selected)
    query_id = str(uuid.uuid4())
    trace = {"query_id": query_id, "original_query": question, "rewritten_query": plan.normalized, "retrieval_method": "multi_query_hybrid_plus_bridge_plus_small_model", "candidate_count": len(hits), "selected_count": len(selected), "generation_path": generation_path, "repair_used": repair_used, "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}, "pre_repair_claims": [c.to_dict() for c in pre_repair_claims], "claims": [c.to_dict() for c in claims], "grounding": ground, "contradiction_report": contradiction, "firewall_used": firewall_used, "semantic_alignment": semantic_alignment, "clinical_reasoning": reasoning_state, "advanced_reasoning": advanced_reasoning.to_dict(), "semantic_understanding": understanding.to_dict(), "small_model_assist": small_model_assist or {}, "evidence_graph": {"nodes": len(nodes), "edges": len(edges)}}
    try:
        self.state_store.record_query_trace(query_id, sanitize_trace(trace))
    except Exception:
        self.logger.exception("Failed to record query trace")
    self.conversation_memory.add(question, safe_answer)
    return {"query_id": query_id, "status": "SUCCESS" if ground.get("allow") and generation_path != "abstained" and not contradiction.get("has_contradiction") else "SUCCESS_WITH_WARNINGS", "answer": safe_answer, "citations": citations, "hits": list(selected), "confidence": {"level": "high" if evidence_conf >= 0.75 else "medium" if evidence_conf >= 0.50 else "low", "evidence_confidence": evidence_conf}, "query_analysis": plan.to_dict(), "semantic_understanding": understanding.to_dict(), "evidence_alignment": alignment, "semantic_alignment": semantic_alignment, "grounding": ground, "claims": [c.to_dict() for c in claims], "contradiction_report": contradiction, "clinical_reasoning": reasoning_state, "advanced_reasoning": advanced_reasoning.to_dict(), "small_model_assist": small_model_assist or {}, "repair_used": repair_used, "god_mode": True}


def report() -> dict[str, object]:
    return {"name": "GOD_MODE_RAG", "feature_count": len(GOD_MODE_FEATURES), "features": list(GOD_MODE_FEATURES), "extended_capability_count": len(EXTENDED_GOD_MODE_CAPABILITIES), "extended_capabilities": list(EXTENDED_GOD_MODE_CAPABILITIES), "fail_closed": True, "universal_pdf_mode": True, "answer_monkey_patch": False, "composition": "ProductionRAGSystem", "small_model_copilot": True, "critic_repair_loop": True, "post_repair_reverification": True}


def audit_god_mode_index(system: Any) -> dict[str, Any]:
    return audit_index(system)


def install() -> None:
    install_atomic_versioning()
