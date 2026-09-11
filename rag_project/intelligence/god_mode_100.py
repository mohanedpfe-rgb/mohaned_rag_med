"""Authoritative production answer engine.

The production path is intentionally deterministic-first: route the question,
retrieve with multiple complementary queries, rerank with lexical/semantic/
structural signals, self-correct weak retrieval, build a parent-aware evidence
pack, optionally synthesize with the local LLM, and certify every generated claim.
"""
from __future__ import annotations

import re
import time
from typing import Any

from rag_project.intelligence.advanced_rag_engine import (
    build_answer_plan,
    retrieve_document_aware,
)
from rag_project.intelligence.adaptive_retrieval import choose_retrieval_budget
from rag_project.intelligence.confidence_calibration import calibrate_confidence
from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix
from rag_project.intelligence.entity_coverage import score_entity_coverage
from rag_project.intelligence.hierarchical_evidence import build_evidence_hierarchy, select_context_levels
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.utils.text_utils import meaningful_tokens

PIPELINE_AUTHORITY = "rag_project.intelligence.god_mode_100.enhanced_god_answer"


def _claim_texts(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("claim", ""))
        for item in result.get("claims", [])
        if isinstance(item, dict) and str(item.get("claim", "")).strip()
    ]


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _validated_model_entities(result: dict[str, Any]) -> list[str]:
    assist = result.get("small_model_assist") or {}
    raw = [str(x).strip() for x in assist.get("entities", []) if str(x).strip()]
    evidence = " ".join(
        str(getattr(hit, "text", "") or "") for hit in (result.get("hits") or [])
    ).casefold()
    return [
        entity for entity in raw
        if meaningful_tokens(entity) and entity.casefold() in evidence
    ][:12]


def _sentence_units(answer: str) -> list[str]:
    rows: list[str] = []
    for raw in re.split(r"\n+|(?<=[.!?؟])\s+", str(answer or "")):
        text = re.sub(r"\s+", " ", raw).strip()
        if text:
            rows.append(re.sub(r"^[-*•\s]+", "", text).strip())
    return rows


def _exact_provenance(answer: str, hits: list[Any]) -> dict[str, Any]:
    """Certify that every [S#] sentence originated in the retrieved evidence."""
    rows: list[tuple[str, int]] = []
    for line in str(answer or "").splitlines():
        match = re.search(r"\[S(\d+)\]\s*$", line, re.I)
        if not match:
            continue
        text = re.sub(r"\s*\[S\d+\]\s*$", "", line, flags=re.I)
        text = re.sub(r"^[-*•\s]+", "", text).strip()
        if text:
            rows.append((text, int(match.group(1))))

    details: list[dict[str, Any]] = []
    matched = 0
    for text, source_no in rows:
        idx = source_no - 1
        if idx < 0 or idx >= len(hits):
            details.append({"claim": text, "source": f"S{source_no}", "matched": False, "reason": "source_out_of_range"})
            continue
        source = re.sub(r"\s+", " ", str(getattr(hits[idx], "text", "") or "")).strip().casefold()
        needle = re.sub(r"\s+", " ", text).strip().casefold()
        ok = len(needle) >= 12 and needle in source
        matched += int(ok)
        details.append({
            "claim": text,
            "source": f"S{source_no}",
            "matched": ok,
            "reason": "exact_source_substring" if ok else "source_mismatch",
        })
    ratio = matched / max(1, len(rows))
    return {"allow": bool(rows) and matched == len(rows), "supported_ratio": ratio, "items": details}


def _extractive_answer(question: str, hits: list[Any], route: dict[str, Any], max_sentences: int) -> tuple[str, list[dict[str, Any]]]:
    q_tokens = set(meaningful_tokens(question))
    stop = {
        "what", "are", "the", "main", "findings", "is", "this", "that", "does",
        "document", "report", "explain", "define", "about", "please", "give", "show",
        "list", "dans", "les", "des", "une", "un", "est", "que", "quels", "quelles",
        "principales", "résultats", "ma", "هو", "هي", "عن", "هذه", "هذا", "ما",
    }
    q_tokens -= stop
    summary = bool(route.get("summary"))
    ranked: list[tuple[float, str, int]] = []
    for source_index, hit in enumerate(hits, start=1):
        text = str(getattr(hit, "text", "") or "")
        base = max(0.0, min(1.0, float(getattr(hit, "score", 0.0) or 0.0)))
        for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", text):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            sentence = re.sub(r"^\[(?:Chapter|Section):[^\]]*\]\s*", "", sentence, flags=re.I).strip()
            if len(sentence) < 25:
                continue
            overlap = len(set(meaningful_tokens(sentence)) & q_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
            structural = 0.10 if summary and re.search(
                r"\b(introduction|overview|conclusion|diagnosis|etiology|clinical|treatment|chapter|section|rappel|exploration|diagnostic|traitement|étiologie)\b",
                sentence,
                re.I,
            ) else 0.0
            score = 0.64 * base + 0.30 * overlap + structural
            ranked.append((score, sentence, source_index))
    ranked.sort(key=lambda row: row[0], reverse=True)
    chosen: list[tuple[str, int]] = []
    seen_text: set[str] = set()
    seen_sources: set[int] = set()
    for score, sentence, source_no in ranked:
        key = sentence.casefold()
        if key in seen_text or score < 0.10:
            continue
        if summary and source_no in seen_sources:
            continue
        chosen.append((sentence, source_no))
        seen_text.add(key)
        seen_sources.add(source_no)
        if len(chosen) >= max_sentences:
            break
    answer = "\n".join(f"- {sentence} [S{source_no}]" for sentence, source_no in chosen)
    claims = [
        {"claim": sentence, "source": f"S{source_no}", "provenance": "exact_sentence_from_retrieved_chunk"}
        for sentence, source_no in chosen
    ]
    return answer, claims


def _synthesize_from_evidence(system: Any, question: str, evidence_bundle: str) -> str | None:
    llm = getattr(system, "llm", None)
    if llm is None or not evidence_bundle.strip():
        return None
    prompt = (
        "You are a medical study assistant operating in strict document-grounded mode. "
        "Answer ONLY from the evidence below. Every factual sentence MUST end with one or more "
        "citation markers that already exist in the evidence, such as [S1]. Do not invent facts, "
        "numbers, diagnoses, recommendations, causes, or qualifiers. Preserve negation and units. "
        "When the evidence is incomplete, state the limitation instead of guessing. Return only the answer.\n\n"
        f"Question:\n{question[:2500]}\n\nEvidence:\n{evidence_bundle[:12000]}"
    )
    try:
        value = llm.generate(
            prompt=prompt,
            system_prompt="Use the supplied book evidence as the only factual authority.",
            temperature=0.0,
        )
        result = str(value or "").strip()
        return result[:10000] if result else None
    except Exception:
        return None


def enhance_result(system: Any, question: str, result: dict[str, Any], metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    """Non-authoritative diagnostics; never erase or replace a valid answer."""
    if not isinstance(result, dict) or not result:
        return result
    enhanced = dict(result)
    try:
        hits = list(result.get("hits") or [])
        plan = _safe_dict(result.get("query_analysis"))
        understanding = _safe_dict(result.get("semantic_understanding"))
        budget = choose_retrieval_budget(
            query_tokens=len(meaningful_tokens(question)),
            entity_count=len(plan.get("entities") or ()),
            intent=str(plan.get("intent", "")),
            confidence=float(understanding.get("confidence", 0.0) or 0.0),
            initial_score=float((_safe_dict(result.get("semantic_alignment"))).get("score", 0.0) or 0.0),
        )
        claims = _claim_texts(result)
        matrix = build_claim_evidence_matrix(
            claims, hits, [f"S{i + 1}" for i in range(len(hits))]
        ) if claims and hits else ()
        hierarchy = build_evidence_hierarchy(hits)
        enhanced.update({
            "evidence_claim_matrix": [record.to_dict() for record in matrix],
            "evidence_hierarchy": [record.to_dict() for record in hierarchy[:100]],
            "evidence_context_levels": select_context_levels(hierarchy),
            "adaptive_retrieval_budget": budget.to_dict(),
            "pipeline_authority": PIPELINE_AUTHORITY,
            "validated_small_model_entities": _validated_model_entities(result),
        })
        retrieval_score = max((float(getattr(h, "score", 0.0) or 0.0) for h in hits), default=0.0)
        retrieval_meta = _safe_dict(result.get("retrieval_quality"))
        entailment = float((_safe_dict(result.get("grounding"))).get("supported_ratio", 0.0) or 0.0)
        coverage = float(retrieval_meta.get("evidence_coverage", 0.0) or 0.0)
        calibration = calibrate_confidence(
            retrieval=min(1.0, retrieval_score),
            rerank=min(1.0, retrieval_score),
            entailment=entailment,
            entity_coverage=float((_safe_dict(result.get("entity_coverage"))).get("coverage", 1.0) or 1.0),
            source_agreement=float((_safe_dict(result.get("contradiction_report"))).get("agreement", 1.0) or 1.0),
            contradiction=1.0 if (_safe_dict(result.get("contradiction_report"))).get("has_contradiction") else 0.0,
            safety_conflict=float((_safe_dict(result.get("advanced_reasoning"))).get("safety_conflict", 0.0) or 0.0),
        ).to_dict()
        calibration["coverage"] = coverage
        enhanced["confidence_calibration"] = calibration
        enhanced["confidence"] = {"level": calibration.get("level", "low"), "evidence_confidence": calibration.get("calibrated", 0.0)}
    except Exception:
        pass
    return enhanced


def enhanced_god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical production answer path."""
    started = time.perf_counter()
    clean_question = re.sub(r"\s+", " ", str(question or "")).strip()[:3500]
    if not clean_question:
        return {
            "status": "LOW_QUALITY_QUERY",
            "answer": "Please provide a precise question.",
            "citations": [], "hits": [],
            "confidence": {"level": "none", "evidence_confidence": 0.0},
            "pipeline_authority": PIPELINE_AUTHORITY,
        }

    settings = getattr(self, "settings", None)
    top_k = max(4, min(12, int(getattr(settings, "top_k", 8)))) if settings is not None else 8

    # ---- Phase 1/2: route + retrieval -------------------------------------------------
    try:
        hits, retrieval_state = retrieve_document_aware(
            self, clean_question, metadata_filter, top_k=top_k
        )
    except Exception as exc:
        return {
            "status": "ANSWER_UNAVAILABLE",
            "answer": "The indexed documents could not be searched safely right now.",
            "citations": [], "hits": [],
            "confidence": {"level": "none", "evidence_confidence": 0.0},
            "recovery": {"retrieval_error": type(exc).__name__},
            "pipeline_authority": PIPELINE_AUTHORITY,
        }

    route = _safe_dict(retrieval_state.get("route"))
    coverage = _safe_dict(retrieval_state.get("coverage"))
    answer_plan = build_answer_plan(clean_question, route_query_from_state(route), coverage)

    if not hits:
        return {
            "status": "NOT_SUPPORTED",
            "answer": "I could not find sufficient evidence in the indexed documents to answer this question.",
            "citations": [], "hits": [],
            "confidence": {"level": "none", "evidence_confidence": 0.0},
            "pipeline_authority": PIPELINE_AUTHORITY,
            "query_trace": {
                "mode": "document_aware",
                "question": clean_question,
                "routing": route,
                "retrieval": retrieval_state,
                "generation": {"status": "not_attempted"},
                "verification": {"status": "not_attempted"},
            },
        }

    # ---- Phase 3: evidence planning / deterministic answer ---------------------------
    max_sentences = 10 if route.get("summary") else 8
    answer, provenance_claims = _extractive_answer(clean_question, hits, route, max_sentences)
    if not answer:
        return {
            "status": "NOT_SUPPORTED",
            "answer": "Relevant chunks were retrieved, but no usable evidence sentence could be selected.",
            "citations": [], "hits": hits,
            "confidence": {"level": "low", "evidence_confidence": 0.0},
            "pipeline_authority": PIPELINE_AUTHORITY,
        }

    provenance = _exact_provenance(answer, hits)
    generation_path = "deterministic_extractive"
    evidence_bundle = "\n".join(
        f"[S{i + 1}] {str(getattr(hit, 'text', '') or '')[:2600]}"
        for i, hit in enumerate(hits[:max(top_k * 3, 12)])
    )

    # LLM synthesis is optional. It can only replace the deterministic draft after
    # semantic claim verification succeeds; otherwise the certified extractive draft wins.
    if coverage.get("sufficient", False):
        synthesized = _synthesize_from_evidence(self, clean_question, evidence_bundle)
        if synthesized:
            try:
                from rag_project.intelligence.evidence_guard import verify_claims, grounding_decision
                candidate_hits = hits[:max(top_k * 3, 12)]
                claims = list(verify_claims(
                    synthesized,
                    [str(getattr(h, "text", "") or "") for h in candidate_hits],
                    [f"S{i + 1}" for i in range(len(candidate_hits))],
                ))
                ground = grounding_decision(claims, min_supported_ratio=0.70) if claims else {"allow": False, "supported_ratio": 0.0}
                citation_markers = re.findall(r"\[S\d+\]", synthesized)
                all_cited = bool(citation_markers) and len(citation_markers) >= max(1, len(_sentence_units(synthesized)))
                if claims and ground.get("allow") and all_cited:
                    answer = synthesized
                    provenance_claims = [claim.to_dict() for claim in claims]
                    generation_path = "local_llm_grounded"
            except Exception:
                pass

    # ---- Phase 4/5: citation + certification -----------------------------------------
    selected = hits[:max(top_k * 3, 12)]
    citations: list[Any] = []
    try:
        built = self.citation_manager.build(selected) or []
        citations = self.citation_manager.validate(built, selected) or []
    except Exception:
        citations = []

    if generation_path == "deterministic_extractive":
        verified = _exact_provenance(answer, selected)
        grounding = {
            "allow": bool(verified.get("allow")),
            "supported_ratio": float(verified.get("supported_ratio", 0.0) or 0.0),
            "method": "exact_retrieved_sentence_provenance",
            "verified_items": verified.get("items", []),
        }
    else:
        try:
            from rag_project.intelligence.evidence_guard import verify_claims, grounding_decision
            claims = list(verify_claims(
                answer,
                [str(getattr(h, "text", "") or "") for h in selected],
                [f"S{i + 1}" for i in range(len(selected))],
            ))
            grounding = dict(grounding_decision(claims, min_supported_ratio=0.70) if claims else {"allow": False, "supported_ratio": 0.0})
            grounding["method"] = "semantic_claim_verification"
            if claims:
                provenance_claims = [claim.to_dict() for claim in claims]
        except Exception:
            grounding = {"allow": False, "supported_ratio": 0.0, "method": "verification_error"}

    if not grounding.get("allow"):
        return {
            "status": "ANSWER_UNAVAILABLE",
            "answer": "The evidence was retrieved, but the answer could not be certified as sufficiently grounded.",
            "citations": [],
            "hits": selected,
            "confidence": {"level": "low", "evidence_confidence": float(grounding.get("supported_ratio", 0.0) or 0.0)},
            "grounding": grounding,
            "claims": provenance_claims,
            "retrieval_quality": {
                "evidence_coverage": float(coverage.get("overall", 0.0) or 0.0),
                "entity_coverage": float(coverage.get("entity_coverage", 0.0) or 0.0),
            },
            "answer_plan": answer_plan,
            "query_trace": {
                "mode": "document_aware",
                "question": clean_question,
                "routing": route,
                "retrieval": retrieval_state,
                "generation": {"status": "completed", "path": generation_path},
                "verification": {"status": "failed", "method": grounding.get("method"), "supported_ratio": grounding.get("supported_ratio", 0.0)},
                "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 2)},
            },
            "pipeline_authority": PIPELINE_AUTHORITY,
        }

    # Optional deterministic planner metadata is diagnostic; it cannot block the answer.
    try:
        deterministic = __import__(
            "rag_project.intelligence.top_level_pipeline", fromlist=["deterministic_phase1"]
        ).deterministic_phase1(clean_question, conversation_context="")
        phase_plan = deterministic.to_dict()
    except Exception:
        phase_plan = {"intent": route.get("kind", "factual"), "entities": list(route.get("entities") or ())}
    try:
        entity_report = score_entity_coverage(
            clean_question, selected, planned_entities=phase_plan.get("entities") or ()
        )
    except Exception:
        entity_report = {"coverage": 1.0, "covered": [], "missing": [], "query_entities": []}

    contradiction_report = retrieval_state.get("contradiction") or {"has_contradiction": False, "conflicts": []}
    status = "SUCCESS_WITH_WARNINGS" if contradiction_report.get("has_contradiction") or not citations else "SUCCESS"
    result = {
        "query_id": f"bookrag-{int(time.time() * 1000)}",
        "status": status,
        "answer": answer,
        "citations": citations,
        "hits": selected,
        "confidence": {
            "level": "high" if grounding.get("supported_ratio", 0.0) >= 0.85 else "medium",
            "evidence_confidence": float(grounding.get("supported_ratio", 0.0) or 0.0),
        },
        "grounding": grounding,
        "claims": provenance_claims,
        "query_analysis": phase_plan,
        "rewritten_question": clean_question,
        "phase_plan": phase_plan,
        "answer_plan": answer_plan,
        "entity_coverage": entity_report,
        "retrieval_quality": {
            "evidence_coverage": float(coverage.get("overall", 0.0) or 0.0),
            "entity_coverage": float(coverage.get("entity_coverage", 0.0) or 0.0),
            "candidate_count": int(retrieval_state.get("candidates", 0) or 0),
            "final_hits": int(retrieval_state.get("final_hits", len(selected)) or len(selected)),
            "self_corrections": int(retrieval_state.get("self_corrections", 0) or 0),
        },
        "contradiction_report": contradiction_report,
        "generation_path": generation_path,
        "god_mode": True,
        "evidence_first": True,
        "document_aware": True,
        "pipeline_authority": PIPELINE_AUTHORITY,
        "query_trace": {
            "mode": "document_aware",
            "question": clean_question,
            "routing": route,
            "retrieval": retrieval_state,
            "generation": {"status": "completed", "path": generation_path},
            "verification": {
                "status": "completed",
                "method": grounding.get("method"),
                "supported_ratio": grounding.get("supported_ratio", 0.0),
            },
            "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 2)},
        },
    }
    return enhance_result(self, clean_question, result, metadata_filter)


def route_query_from_state(state: dict[str, Any]):
    """Convert a serialized route dictionary into the small interface used by planning."""
    class _Route:
        def __init__(self, value: dict[str, Any]):
            self.kind = str(value.get("kind", "factual"))
            self.scope = str(value.get("scope", "question"))
            self.needs_table = bool(value.get("needs_table", False))
            self.needs_numeric = bool(value.get("needs_numeric", False))
            self.needs_figure = bool(value.get("needs_figure", False))
            self.multi_hop = bool(value.get("multi_hop", False))
            self.summary = bool(value.get("summary", False))
            self.comparison = bool(value.get("comparison", False))
            self.exact_lookup = bool(value.get("exact_lookup", False))
            self.expected_slots = tuple(value.get("expected_slots", ()))
            self.entities = tuple(value.get("entities", ()))
            self.confidence = float(value.get("confidence", 0.8))
    return _Route(state)


def legacy_enhanced_god_answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    from rag_project.intelligence.god_mode import _god_answer
    base = _god_answer(self, question, metadata_filter)
    return enhance_result(self, question, base, metadata_filter)


__all__ = ["enhanced_god_answer", "enhance_result", "legacy_enhanced_god_answer"]
