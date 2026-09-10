from __future__ import annotations

from typing import Any

from rag_project.intelligence.adaptive_retrieval import choose_retrieval_budget
from rag_project.intelligence.confidence_calibration import calibrate_confidence
from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix
from rag_project.intelligence.entity_coverage import score_entity_coverage
from rag_project.intelligence.final_answer_contract import verify_final_answer
from rag_project.intelligence.hierarchical_evidence import build_evidence_hierarchy, select_context_levels
from rag_project.intelligence.top_level_pipeline import complete_phases
from rag_project.utils.text_utils import meaningful_tokens

PIPELINE_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"


def _claim_texts(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("claim", ""))
        for item in result.get("claims", [])
        if isinstance(item, dict) and str(item.get("claim", "")).strip()
    ]


def _validated_model_entities(result: dict[str, Any]) -> list[str]:
    assist = result.get("small_model_assist") or {}
    raw = [str(x).strip() for x in assist.get("entities", []) if str(x).strip()]
    evidence = " ".join(
        str(getattr(hit, "text", "") or "") for hit in (result.get("hits") or [])
    ).casefold()
    return [
        entity
        for entity in raw
        if meaningful_tokens(entity) and entity.casefold() in evidence
    ][:12]


def _runtime_phase_implementation(
    completed: dict[str, Any], final_matrix: list[Any], final_verification: dict[str, Any]
) -> dict[str, Any]:
    phases = completed.get("phases") or {}
    plan = completed.get("phase_plan") or {}
    retrieval = completed.get("adaptive_retrieval") or {}
    two_stage = completed.get("two_stage_synthesis") or {}
    blocked = int(final_verification.get("blocked_claims", 0))
    p3_status = (
        "synthesized"
        if two_stage.get("used")
        else "required_abstention"
        if two_stage.get("required")
        else "extractive_verified_fallback"
    )
    return {
        "phase_1_query_understanding": {
            "status": phases.get("phase_1_query_understanding", "unknown"),
            "planner_source": plan.get("planner_source", "unknown"),
            "planner_confidence": plan.get("planner_confidence", 0.0),
            "entity_count": len(plan.get("entities") or ()),
        },
        "phase_2_retrieval_precision": {
            "status": phases.get("phase_2_retrieval_precision", "unknown"),
            "stage": retrieval.get("stage", 0),
            "queries": retrieval.get("queries", 0),
            "final_hits": retrieval.get("final_hits", 0),
            "escalated": bool(retrieval.get("escalated")),
        },
        "phase_3_two_stage_generation": {
            "status": p3_status,
            "required": bool(two_stage.get("required")),
            "attempted": bool(two_stage.get("attempted")),
            "used": bool(two_stage.get("used")),
            "fallback": bool(two_stage.get("fallback")),
            "verified": bool((two_stage.get("verification") or {}).get("checked"))
            and not bool((two_stage.get("verification") or {}).get("blocked")),
        },
        "phase_4_verification": {
            "status": phases.get("phase_4_verification", "unknown"),
            "claim_count": len(final_matrix),
            "blocked_claims": blocked,
            "hard_gate": True,
            "final_answer_checked": bool(final_verification.get("checked")),
            "calibrated": bool(completed.get("confidence_calibration")),
        },
        "phase_5_intelligence_visibility": {
            "status": phases.get("phase_5_intelligence_visibility", "unknown"),
            "signals_present": all(
                key in completed
                for key in (
                    "phase_plan",
                    "rewritten_question",
                    "evidence_claim_matrix",
                    "confidence_calibration",
                    "entity_coverage",
                    "final_verification",
                )
            ),
            "authority": PIPELINE_AUTHORITY,
        },
    }


def enhance_result(
    system: Any,
    question: str,
    result: dict[str, Any],
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete and certify all five phases; only the final verified answer may escape."""
    if not result:
        return result

    hits = list(result.get("hits") or [])
    source_ids = [f"S{i + 1}" for i in range(len(hits))]
    claims = _claim_texts(result)
    matrix = build_claim_evidence_matrix(claims, hits, source_ids) if claims and hits else ()
    hierarchy = build_evidence_hierarchy(hits)
    context_levels = select_context_levels(hierarchy)
    understanding = result.get("semantic_understanding") or {}
    query_plan = result.get("query_analysis") or {}
    budget = choose_retrieval_budget(
        query_tokens=len(meaningful_tokens(question)),
        entity_count=len(query_plan.get("entities", ()) or ()),
        intent=str(query_plan.get("intent", "")),
        confidence=float(understanding.get("confidence", 0.0) or 0.0),
        initial_score=float(result.get("semantic_alignment", {}).get("score", 0.0) or 0.0),
    )
    entailment = (
        sum(record.support for record in matrix) / max(1, len(matrix))
        if matrix
        else float(result.get("grounding", {}).get("supported_ratio", 0.0) or 0.0)
    )
    entity_coverage = float(
        result.get("advanced_reasoning", {}).get(
            "entity_coverage",
            result.get("semantic_alignment", {}).get("entity_coverage", 0.0),
        )
        or 0.0
    )
    source_agreement = float(result.get("advanced_reasoning", {}).get("source_agreement", 0.0) or 0.0)
    contradiction = 1.0 if (result.get("contradiction_report") or {}).get("has_contradiction") else 0.0
    safety_conflict = float(result.get("advanced_reasoning", {}).get("safety_conflict", 0.0) or 0.0)
    selected_scores = [float(getattr(hit, "score", 0.0)) for hit in hits]
    top_score = max(selected_scores, default=0.0)

    calibration = calibrate_confidence(
        retrieval=top_score,
        rerank=top_score,
        entailment=entailment,
        entity_coverage=(1.0 if not query_plan.get("entities") else entity_coverage),
        source_agreement=source_agreement,
        contradiction=contradiction,
        safety_conflict=safety_conflict,
    )
    enhanced = dict(result)
    enhanced["evidence_claim_matrix"] = [record.to_dict() for record in matrix]
    enhanced["evidence_hierarchy"] = [record.to_dict() for record in hierarchy[:80]]
    enhanced["evidence_context_levels"] = context_levels
    enhanced["adaptive_retrieval_budget"] = budget.to_dict()
    enhanced["validated_small_model_entities"] = _validated_model_entities(result)
    enhanced["confidence_calibration"] = calibration.to_dict()
    enhanced["confidence"] = {
        "level": calibration.level,
        "evidence_confidence": calibration.calibrated,
    }
    enhanced["pipeline_authority"] = PIPELINE_AUTHORITY

    completed = complete_phases(system, question, enhanced, metadata_filter)
    final_hits = list(completed.get("hits") or hits)
    phase_plan = completed.get("phase_plan") or {}
    final_matrix = ()

    provisional_answer = str(completed.get("answer") or "").strip()
    final_verification = verify_final_answer(
        provisional_answer,
        final_hits,
        require_entailment=bool(phase_plan.get("needs_numeric"))
        or str(phase_plan.get("intent", "")) in {"diagnosis", "management", "etiology", "mechanism", "prognosis"},
    )
    final_matrix = final_verification.get("evidence_claim_matrix") or []
    completed["evidence_claim_matrix"] = final_matrix
    completed["final_evidence_claim_matrix"] = final_matrix
    completed["final_claim_checks"] = final_verification.get("claim_checks", [])
    completed["final_verification"] = final_verification
    completed["pipeline_authority"] = PIPELINE_AUTHORITY

    entity_report = score_entity_coverage(
        str(completed.get("rewritten_question") or question),
        final_hits,
        planned_entities=phase_plan.get("entities") or (),
    )
    completed["entity_coverage"] = entity_report

    hard_risk = bool(phase_plan.get("needs_numeric")) or str(phase_plan.get("intent", "")) in {
        "diagnosis",
        "management",
        "etiology",
        "mechanism",
        "prognosis",
    }
    final_allowed = bool(final_verification.get("allow"))
    if not final_allowed or (hard_risk and not bool(final_verification.get("matrix_all_entailed"))):
        reasons = []
        if final_verification.get("reason"):
            reasons.append(str(final_verification["reason"]))
        if entity_report.get("missing") and entity_report.get("entity_count", 0) > 0:
            reasons.append("missing_query_entities_in_evidence")
        completed["status"] = "REASONING_ABSTAIN"
        completed["abstained"] = True
        completed["abstention_reasons"] = reasons or ["final_answer_verification_failed"]
        completed["answer"] = (
            "I could not verify a sufficiently grounded answer from the indexed evidence; "
            "unsupported or conflicting clinical details were withheld."
        )
        completed["citations"] = []

    final_entailment = float(final_verification.get("supported_ratio", 0.0) or 0.0)
    completed_calibration = calibrate_confidence(
        retrieval=top_score,
        rerank=top_score,
        entailment=final_entailment,
        entity_coverage=(1.0 if entity_report.get("entity_count", 0) == 0 else max(entity_report.get("coverage", 0.0), entity_report.get("partial_coverage", 0.0) * 0.75)),
        source_agreement=source_agreement,
        contradiction=contradiction,
        safety_conflict=safety_conflict,
    )
    if not final_allowed:
        completed_calibration = calibrate_confidence(
            retrieval=top_score,
            rerank=top_score,
            entailment=0.0,
            entity_coverage=(1.0 if entity_report.get("entity_count", 0) == 0 else 0.0),
            source_agreement=source_agreement,
            contradiction=max(contradiction, 1.0),
            safety_conflict=max(safety_conflict, 0.0),
        )
    completed["confidence_calibration"] = completed_calibration.to_dict()
    completed["confidence"] = {
        "level": completed_calibration.level,
        "evidence_confidence": completed_calibration.calibrated,
    }
    completed["phase_implementation"] = _runtime_phase_implementation(
        completed, final_matrix, final_verification
    )
    completed["phase_implementation"]["hardware_profile"] = "llama3.2:3b + nomic-embed-text + i5/16GB"
    completed["phase_implementation"]["final_claim_evidence_hard_gate"] = True
    completed["phase_implementation"]["final_confidence_is_calibrated"] = True
    completed["phase_implementation"]["canonical_pipeline_executed"] = True
    completed["phase_implementation"]["pipeline_authority"] = PIPELINE_AUTHORITY
    completed["god_mode_100"] = True
    return completed


def enhanced_god_answer(
    self: Any,
    question: str,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from rag_project.intelligence.god_mode import _god_answer

    base = _god_answer(self, question, metadata_filter)
    return enhance_result(self, question, base, metadata_filter)
