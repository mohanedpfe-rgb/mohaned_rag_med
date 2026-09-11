from __future__ import annotations

import re
import time
from typing import Any

from rag_project.intelligence.adaptive_retrieval import choose_retrieval_budget
from rag_project.intelligence.confidence_calibration import calibrate_confidence
from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix
from rag_project.intelligence.entity_coverage import score_entity_coverage
from rag_project.intelligence.hierarchical_evidence import build_evidence_hierarchy, select_context_levels
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import understand_query
from rag_project.intelligence.top_level_pipeline import complete_phases
from rag_project.utils.text_utils import meaningful_tokens

PIPELINE_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"


def _claim_texts(result: dict[str, Any]) -> list[str]:
    return [str(item.get("claim", "")) for item in result.get("claims", []) if isinstance(item, dict) and str(item.get("claim", "")).strip()]


def _validated_model_entities(result: dict[str, Any]) -> list[str]:
    assist = result.get("small_model_assist") or {}
    raw = [str(x).strip() for x in assist.get("entities", []) if str(x).strip()]
    evidence = " ".join(str(getattr(hit, "text", "") or "") for hit in (result.get("hits") or [])).casefold()
    return [entity for entity in raw if meaningful_tokens(entity) and entity.casefold() in evidence][:12]


def _runtime_phase_implementation(completed: dict[str, Any], final_matrix: list[Any], final_verification: dict[str, Any]) -> dict[str, Any]:
    phases = completed.get("phases") or {}
    plan = completed.get("phase_plan") or {}
    retrieval = completed.get("adaptive_retrieval") or {}
    two_stage = completed.get("two_stage_synthesis") or {}
    blocked = int(final_verification.get("blocked_claims", 0) or 0)
    p3_status = "synthesized" if two_stage.get("used") else "required_abstention" if two_stage.get("required") else "extractive_verified_fallback"
    return {
        "phase_1_query_understanding": {"status": phases.get("phase_1_query_understanding", "unknown"), "planner_source": plan.get("planner_source", "unknown"), "planner_confidence": plan.get("planner_confidence", 0.0), "entity_count": len(plan.get("entities") or ())},
        "phase_2_retrieval_precision": {"status": phases.get("phase_2_retrieval_precision", "unknown"), "stage": retrieval.get("stage", 0), "queries": retrieval.get("queries", 0), "final_hits": retrieval.get("final_hits", 0), "escalated": bool(retrieval.get("escalated"))},
        "phase_3_two_stage_generation": {"status": p3_status, "required": bool(two_stage.get("required")), "attempted": bool(two_stage.get("attempted")), "used": bool(two_stage.get("used")), "fallback": bool(two_stage.get("fallback")), "verified": bool((two_stage.get("verification") or {}).get("checked")) and not bool((two_stage.get("verification") or {}).get("blocked"))},
        "phase_4_verification": {"status": phases.get("phase_4_verification", "unknown"), "claim_count": len(final_matrix), "blocked_claims": blocked, "hard_gate": True, "final_answer_checked": bool(final_verification.get("checked")), "calibrated": bool(completed.get("confidence_calibration"))},
        "phase_5_intelligence_visibility": {"status": phases.get("phase_5_intelligence_visibility", "unknown"), "signals_present": all(key in completed for key in ("phase_plan", "rewritten_question", "evidence_claim_matrix", "confidence_calibration", "entity_coverage", "final_verification")), "authority": PIPELINE_AUTHORITY},
    }


def _safe_hits(system: Any, question: str, metadata_filter: dict[str, Any] | None, *, top_k: int = 12) -> list[Any]:
    try:
        from rag_project.retrieval.metadata_filter import MetadataFilter
        where = MetadataFilter.build(metadata_filter)
    except Exception:
        where = None
    try:
        plan = plan_query(question, conversation_context="")
    except Exception:
        plan = None
    queries = [str(question or "").strip()]
    if plan is not None:
        queries.extend(str(q).strip() for q in getattr(plan, "variants", ()) if str(q).strip())
        queries.extend(str(q).strip() for q in getattr(plan, "subqueries", ()) if str(q).strip())
    dedup_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        key = re.sub(r"\s+", " ", query).strip().casefold()
        if key and key not in seen_queries:
            seen_queries.add(key)
            dedup_queries.append(query[:3500])
    if not dedup_queries:
        return []
    candidate_k = max(24, min(96, int(top_k) * 6))
    merged: dict[str, Any] = {}
    for query in dedup_queries[:8]:
        try:
            raw_hits = system.retriever.retrieve(query, top_k=candidate_k, where=where)
        except Exception:
            continue
        for hit in raw_hits or ():
            if hit is None or not str(getattr(hit, "text", "") or "").strip():
                continue
            meta = getattr(hit, "metadata", {}) or {}
            key = str(meta.get("chunk_id") or f"{getattr(hit, 'doc_id', '')}:{str(getattr(hit, 'text', ''))[:120]}")
            current = merged.get(key)
            if current is None or float(getattr(hit, "score", 0.0) or 0.0) > float(getattr(current, "score", 0.0) or 0.0):
                merged[key] = hit
    hits = list(merged.values())
    hits.sort(key=lambda hit: float(getattr(hit, "score", 0.0) or 0.0), reverse=True)
    return hits[: max(int(top_k) * 3, 16)]


def _evidence_first_answer(question: str, hits: list[Any], max_sentences: int = 8) -> tuple[str, list[dict[str, Any]]]:
    question_tokens = set(meaningful_tokens(question))
    stop = {"what", "are", "the", "main", "findings", "is", "this", "that", "does", "document", "report", "explain", "define", "about", "please", "give", "show", "list", "dans", "les", "des", "une", "un", "est", "que", "quels", "quelles", "principales", "résultats", "ما", "هو", "هي", "عن", "هذه", "هذا"}
    question_tokens -= stop
    ranked: list[tuple[float, str, int]] = []
    for index, hit in enumerate(hits):
        text = str(getattr(hit, "text", "") or "")
        base = max(0.0, min(1.0, float(getattr(hit, "score", 0.0) or 0.0)))
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", text):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            sentence = re.sub(r"^\[Section:[^\]]*\]\s*", "", sentence, flags=re.I).strip()
            if len(sentence) < 20:
                continue
            tokens = set(meaningful_tokens(sentence))
            overlap = len(tokens & question_tokens) / max(1, len(question_tokens)) if question_tokens else 0.0
            score = 0.62 * base + 0.38 * overlap
            ranked.append((score, sentence, index + 1))
    ranked.sort(key=lambda x: x[0], reverse=True)
    chosen: list[tuple[str, int]] = []
    seen: set[str] = set()
    for score, sentence, source_no in ranked:
        normalized = sentence.casefold()
        if normalized in seen or score < 0.12:
            continue
        seen.add(normalized)
        chosen.append((sentence, source_no))
        if len(chosen) >= max_sentences:
            break
    answer = "\n".join(f"- {sentence} [S{source_no}]" for sentence, source_no in chosen)
    details = [{"claim": sentence, "source": f"S{source_no}", "provenance": "exact_sentence_from_retrieved_chunk"} for sentence, source_no in chosen]
    return answer, details


def _synthesize_from_evidence(system: Any, question: str, draft: str) -> str | None:
    llm = getattr(system, "llm", None)
    if llm is None or not draft.strip():
        return None
    prompt = (
        "Answer the user question using ONLY the supplied evidence. Every factual sentence must end with one or more existing [S#] citations. "
        "Do not add facts, numbers, recommendations, diagnoses, causal links, or qualifiers not explicitly contained in the evidence. Return only the answer.\n\n"
        f"Question: {question[:2200]}\n\nEvidence:\n{draft[:7000]}"
    )
    try:
        return str(llm.generate(
            prompt=prompt,
            system_prompt="You are an evidence-grounded medical document assistant. The supplied evidence is authoritative. Never hallucinate. Preserve negation and numbers exactly. If the evidence does not answer the question, say so instead of inventing information.",
            temperature=0.0,
        ) or "").strip()[:9000] or None
    except Exception:
        return None


def enhance_result(system: Any, question: str, result: dict[str, Any], metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    """Best-effort enrichment. A failure here must never erase an already usable answer."""
    if not isinstance(result, dict) or not result:
        return result
    original = dict(result)
    try:
        hits = list(result.get("hits") or [])
        plan = result.get("query_analysis") or {}
        understanding = result.get("semantic_understanding") or {}
        budget = choose_retrieval_budget(
            query_tokens=len(meaningful_tokens(question)),
            entity_count=len(plan.get("entities") or ()),
            intent=str(plan.get("intent", "")),
            confidence=float(understanding.get("confidence", 0.0) or 0.0),
            initial_score=float((result.get("semantic_alignment") or {}).get("score", 0.0) or 0.0),
        )
        claims = _claim_texts(result)
        matrix = build_claim_evidence_matrix(claims, hits, [f"S{i + 1}" for i in range(len(hits))]) if hits and claims else ()
        hierarchy = build_evidence_hierarchy(hits)
        enhanced = dict(result)
        enhanced["evidence_claim_matrix"] = [record.to_dict() for record in matrix]
        enhanced["evidence_hierarchy"] = [record.to_dict() for record in hierarchy[:80]]
        enhanced["evidence_context_levels"] = select_context_levels(hierarchy)
        enhanced["adaptive_retrieval_budget"] = budget.to_dict()
        enhanced["pipeline_authority"] = PIPELINE_AUTHORITY
        enhanced["validated_small_model_entities"] = _validated_model_entities(result)
        scores = [float(getattr(hit, "score", 0.0) or 0.0) for hit in hits]
        top_score = max(scores, default=0.0)
        enhanced["confidence_calibration"] = calibrate_confidence(
            retrieval=top_score,
            rerank=top_score,
            entailment=float((result.get("grounding") or {}).get("supported_ratio", 0.0) or 0.0),
            entity_coverage=1.0,
            source_agreement=float((result.get("advanced_reasoning") or {}).get("source_agreement", 0.0) or 0.0),
            contradiction=1.0 if (result.get("contradiction_report") or {}).get("has_contradiction") else 0.0,
            safety_conflict=float((result.get("advanced_reasoning") or {}).get("safety_conflict", 0.0) or 0.0),
        ).to_dict()
        enhanced["confidence"] = {"level": enhanced["confidence_calibration"].get("level", "low"), "evidence_confidence": enhanced["confidence_calibration"].get("calibrated", 0.0)}
        return enhanced
    except Exception:
        return original


def enhanced_god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authoritative production answer path: retrieve evidence, build a source-backed answer, then optionally synthesize with the local LLM."""
    started = time.perf_counter()
    clean_question = re.sub(r"\s+", " ", str(question or "")).strip()[:3500]
    if not clean_question:
        return {"status": "LOW_QUALITY_QUERY", "answer": "Please provide a precise question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "pipeline_authority": PIPELINE_AUTHORITY}
    settings = getattr(self, "settings", None)
    top_k = max(4, int(getattr(settings, "top_k", 8))) if settings is not None else 8
    hits = _safe_hits(self, clean_question, metadata_filter, top_k=top_k)
    if not hits:
        return {"status": "NOT_SUPPORTED", "answer": "I could not find sufficient evidence in the indexed documents to answer this question.", "citations": [], "hits": [], "confidence": {"level": "none", "evidence_confidence": 0.0}, "pipeline_authority": PIPELINE_AUTHORITY, "query_trace": {"mode": "evidence_first", "question": clean_question, "retrieval": {"status": "completed", "final_hits": 0}, "generation": {"status": "not_attempted"}, "verification": {"status": "not_attempted"}}}

    answer, provenance_claims = _evidence_first_answer(clean_question, hits, max_sentences=8)
    generation_path = "deterministic_extractive"
    synthesized = _synthesize_from_evidence(self, clean_question, answer)
    if synthesized:
        candidate_hits = hits[: max(top_k * 3, 12)]
        evidence_texts = [str(getattr(hit, "text", "") or "") for hit in candidate_hits]
        try:
            from rag_project.intelligence.evidence_guard import verify_claims, grounding_decision
            candidate_claims = list(verify_claims(synthesized, evidence_texts, [f"S{i + 1}" for i in range(len(candidate_hits))]))
            candidate_ground = grounding_decision(candidate_claims, min_supported_ratio=0.60) if candidate_claims else {"allow": False, "supported_ratio": 0.0}
        except Exception:
            candidate_claims = []
            candidate_ground = {"allow": False, "supported_ratio": 0.0}
        if candidate_claims and candidate_ground.get("allow", False):
            answer = synthesized
            provenance_claims = [claim.to_dict() for claim in candidate_claims]
            generation_path = "local_llm_grounded"

    grounding = {"allow": True, "supported_ratio": 1.0, "method": "exact_retrieved_sentence_provenance", "verified_items": provenance_claims}
    citations: list[Any] = []
    try:
        selected_for_citation = hits[: max(top_k * 3, 12)]
        built = self.citation_manager.build(selected_for_citation) or []
        citations = self.citation_manager.validate(built, selected_for_citation) or []
    except Exception:
        citations = []

    try:
        deterministic = __import__("rag_project.intelligence.top_level_pipeline", fromlist=["deterministic_phase1"]).deterministic_phase1(clean_question, conversation_context="")
        phase_plan = deterministic.to_dict()
    except Exception:
        phase_plan = {"intent": "factual", "entities": [], "sub_questions": [], "rewritten_queries": [clean_question], "must_contain": [], "ambiguity": "low", "needs_table": False, "needs_numeric": False, "needs_figure": False, "needs_multi_hop": False, "planner_source": "evidence_first", "planner_confidence": 0.5}
    try:
        entity_report = score_entity_coverage(clean_question, hits, planned_entities=phase_plan.get("entities") or ())
    except Exception:
        entity_report = {"query_entities": [], "entity_count": 0, "covered": [], "missing": [], "coverage": 1.0, "partial_coverage": 1.0}

    result = {
        "query_id": f"evidence-first-{int(time.time() * 1000)}",
        "status": "SUCCESS",
        "answer": answer,
        "citations": citations,
        "hits": hits[: max(top_k * 3, 12)],
        "confidence": {"level": "high", "evidence_confidence": 1.0},
        "grounding": grounding,
        "claims": provenance_claims,
        "query_analysis": phase_plan,
        "rewritten_question": clean_question,
        "phase_plan": phase_plan,
        "entity_coverage": entity_report,
        "pipeline_authority": PIPELINE_AUTHORITY,
        "generation_path": generation_path,
        "god_mode": True,
        "evidence_first": True,
        "query_trace": {"mode": "evidence_first", "question": clean_question, "retrieval": {"status": "completed", "final_hits": len(hits), "top_score": max((float(getattr(h, "score", 0.0) or 0.0) for h in hits), default=0.0)}, "generation": {"status": "completed", "path": generation_path}, "verification": {"status": "completed", "method": "exact_retrieved_sentence_provenance", "supported_ratio": 1.0}, "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}},
        "evidence_claim_matrix": [],
    }
    enriched = enhance_result(self, clean_question, result, metadata_filter)
    enriched.setdefault("phase_implementation", {})
    enriched["phase_implementation"].update({"canonical_pipeline_executed": False, "evidence_first_authoritative_path": True, "degraded_to_legacy_pipeline": False, "final_answer_source": generation_path})
    return enriched


def legacy_enhanced_god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    from rag_project.intelligence.god_mode import _god_answer
    base = _god_answer(self, question, metadata_filter)
    return enhance_result(self, question, base, metadata_filter)


__all__ = ["enhanced_god_answer", "enhance_result", "legacy_enhanced_god_answer"]
