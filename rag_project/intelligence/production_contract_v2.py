"""Canonical request/evidence/answer contract for the production RAG path.

This module is deliberately pure: it defines and applies data contracts but
never imports or monkey-patches the legacy application, top_level_pipeline, or
God-mode compatibility layers.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from rag_project.intelligence.pipeline_integrity import is_control_message, safe_extract_query_entities
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import understand_query
from rag_project.utils.text_utils import meaningful_tokens

CONTRACT_VERSION = "2026-09-14-production-contract-v3"
_HARD_INTENTS = {"comparison", "etiology", "mechanism", "diagnosis", "management", "prognosis", "causal", "relationship"}
_HARD_CUES = (
    "why", "how does", "how do", "compare", "versus", " vs ", "difference", "contraindication",
    "dose", "dosage", "treatment", "management", "diagnosis", "prognosis", "mechanism",
    "pourquoi", "comment", "comparaison", "différence", "traitement", "diagnostic", "pronostic",
    "مقارنة", "لماذا", "كيف", "جرعة", "علاج", "تشخيص",
)
_FOLLOWUP_CUES = re.compile(
    r"(?:^|\s)(?:it|this|that|they|them|those|these|what about|how about|the latter|the former|"
    r"and this|and that|ça|cela|celui|celle|et le|et la|puis|et ça|هذا|هذه|ذلك|تلك|ثم|و)(?:$|\s|[?.!,;:])",
    re.I | re.UNICODE,
)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    original_question: str
    canonical_question: str
    is_followup: bool
    conversation_used: bool
    intent: str
    entities: tuple[str, ...]
    sub_questions: tuple[str, ...]
    query_variants: tuple[str, ...]
    ambiguity: str
    complexity: str
    needs_numeric: bool
    needs_table: bool
    needs_figure: bool
    needs_multi_hop: bool
    metadata_filter: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    document_id: str
    file_name: str
    pages: tuple[Any, ...]
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBundle:
    request_id: str
    sources: tuple[EvidenceSource, ...]
    hit_count: int
    top_score: float
    mean_score: float
    min_score: float
    entity_coverage: float
    missing_entities: tuple[str, ...]
    contradiction_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceBreakdown:
    retrieval: float
    evidence_quality: float
    entailment: float
    entity_coverage: float
    verification: float
    contradiction: float
    final: float
    level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_request_id() -> str:
    return f"rag-{uuid.uuid4().hex[:16]}"


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _is_followup(question: str, history: Sequence[tuple[str, str]]) -> bool:
    if not history:
        return False
    text = _clean(question)
    if not text:
        return False
    if _FOLLOWUP_CUES.search(text):
        return True
    return bool(re.search(r"\b(it|this|that|them|they|ça|cela|هذا|هذه|ذلك|تلك)\b", text, re.I | re.UNICODE))


def _strict_canonicalize(question: str, history: Sequence[tuple[str, str]]) -> tuple[str, bool]:
    cleaned = _clean(question)
    followup = _is_followup(cleaned, history)
    if not followup:
        return cleaned, False
    previous_questions = [_clean(q) for q, _ in history[-3:] if _clean(q)]
    previous_answers = [_clean(a) for _, a in history[-3:] if _clean(a)]
    anchor = previous_questions[-1] if previous_questions else ""
    if not anchor:
        return cleaned, False
    anchors: list[str] = []
    for text in (anchor, previous_answers[-1] if previous_answers else ""):
        try:
            understanding = understand_query(text)
            anchors.extend(getattr(item, "normalized", "") for item in understanding.entities[:8])
        except Exception:
            continue
    unique = tuple(dict.fromkeys(_clean(value).casefold() for value in anchors if _clean(value)))
    if unique and any(anchor_text in cleaned.casefold() for anchor_text in unique):
        return cleaned[:3500], True
    if unique:
        return _clean(f"{cleaned} {' '.join(unique)}")[:3500], True
    if anchor.casefold() in cleaned.casefold():
        return cleaned[:3500], True
    return _clean(f"{anchor} {cleaned}")[:3500], True


def _complexity(plan: Any, understanding: Any, question: str) -> str:
    intent = str(getattr(plan, "intent", "other") or "other").casefold()
    normalized = f" {_clean(question).casefold()} "
    hard = bool(
        getattr(plan, "needs_multi_hop", False)
        or getattr(plan, "needs_numeric", False)
        or getattr(plan, "needs_table", False)
        or getattr(plan, "needs_figure", False)
        or getattr(understanding, "relations", ())
        or len(getattr(plan, "subqueries", ()) or ()) > 1
        or intent in _HARD_INTENTS
        or any(cue in normalized for cue in _HARD_CUES)
    )
    if hard:
        return "hard"
    return "moderate" if len(meaningful_tokens(question)) >= 12 else "simple"


def build_request_context(
    question: str,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    request_id: str | None = None,
) -> RequestContext:
    prior = tuple(history or ())
    original = _clean(question)
    canonical, followup = _strict_canonicalize(original, prior)
    understanding = understand_query(canonical)
    plan = plan_query(canonical)
    entities = safe_extract_query_entities(canonical, getattr(plan, "entities", ()) or ())
    sub_questions = tuple(_clean(value) for value in (getattr(plan, "subqueries", ()) or ()) if _clean(value))[:8]
    variants = tuple(dict.fromkeys(
        _clean(value)
        for value in (canonical, *(getattr(plan, "variants", ()) or ()), getattr(plan, "normalized", ""))
        if _clean(value)
    ))[:8]
    return RequestContext(
        request_id=request_id or new_request_id(),
        original_question=original,
        canonical_question=canonical,
        is_followup=followup,
        conversation_used=followup,
        intent=str(getattr(plan, "intent", "other") or "other"),
        entities=tuple(entities),
        sub_questions=sub_questions,
        query_variants=variants,
        ambiguity=str(getattr(understanding, "ambiguity", "low") or "low"),
        complexity=_complexity(plan, understanding, canonical),
        needs_numeric=bool(getattr(plan, "needs_numeric", False)),
        needs_table=bool(getattr(plan, "needs_table", False)),
        needs_figure=bool(getattr(plan, "needs_figure", False)),
        needs_multi_hop=bool(getattr(plan, "needs_multi_hop", False)),
        metadata_filter=dict(metadata_filter or {}),
    )


def build_evidence_bundle(
    request_id: str,
    hits: Sequence[Any],
    *,
    entity_coverage: Mapping[str, Any] | None = None,
    contradiction_count: int = 0,
) -> EvidenceBundle:
    records: list[EvidenceSource] = []
    for index, hit in enumerate(hits):
        text = _clean(getattr(hit, "text", ""))
        if not text:
            continue
        metadata = getattr(hit, "metadata", {}) or {}
        pages = metadata.get("page_numbers") or metadata.get("pages") or metadata.get("page_number") or ()
        if not isinstance(pages, (list, tuple)):
            pages = (pages,)
        records.append(EvidenceSource(
            source_id=f"S{index + 1}",
            document_id=str(metadata.get("document_id") or getattr(hit, "doc_id", "")),
            file_name=str(metadata.get("file_name") or metadata.get("filename") or ""),
            pages=tuple(pages),
            score=max(0.0, min(1.0, float(getattr(hit, "score", 0.0) or 0.0))),
            text=text,
        ))
    scores = [record.score for record in records]
    report = entity_coverage or {}
    return EvidenceBundle(
        request_id=request_id,
        sources=tuple(records),
        hit_count=len(records),
        top_score=max(scores, default=0.0),
        mean_score=sum(scores) / len(scores) if scores else 0.0,
        min_score=min(scores, default=0.0),
        entity_coverage=max(0.0, min(1.0, float(report.get("coverage", 0.0) or 0.0))),
        missing_entities=tuple(str(value) for value in report.get("missing", ()) if str(value)),
        contradiction_count=max(0, int(contradiction_count)),
    )


def compute_confidence(
    *,
    retrieval: float,
    evidence_quality: float,
    entailment: float,
    entity_coverage: float,
    verification: float,
    contradiction: float,
    hard_gate_passed: bool,
) -> ConfidenceBreakdown:
    values = {
        "retrieval": retrieval,
        "evidence_quality": evidence_quality,
        "entailment": entailment,
        "entity_coverage": entity_coverage,
        "verification": verification,
        "contradiction": contradiction,
    }
    values = {key: max(0.0, min(1.0, float(value))) for key, value in values.items()}
    final = (
        0.20 * values["retrieval"]
        + 0.18 * values["evidence_quality"]
        + 0.24 * values["entailment"]
        + 0.12 * values["entity_coverage"]
        + 0.26 * values["verification"]
        - 0.35 * values["contradiction"]
    )
    final = max(0.0, min(1.0, final))
    if not hard_gate_passed:
        final = min(final, 0.49)
    level = "high" if final >= 0.80 else "medium" if final >= 0.60 else "low"
    return ConfidenceBreakdown(
        values["retrieval"], values["evidence_quality"], values["entailment"],
        values["entity_coverage"], values["verification"], values["contradiction"],
        final, level,
    )


def apply_contract(result: Mapping[str, Any], context: RequestContext) -> dict[str, Any]:
    enhanced = dict(result)
    hits = list(enhanced.get("hits") or ())
    entity_report = enhanced.get("entity_coverage") or {}
    contradiction_report = enhanced.get("contradiction_report") or {}
    final_verification = enhanced.get("final_verification") or {}
    verification_ratio = float(final_verification.get("supported_ratio", 0.0) or 0.0)
    retrieval = max((float(getattr(hit, "score", 0.0) or 0.0) for hit in hits), default=0.0)
    evidence_quality = sum(float(getattr(hit, "score", 0.0) or 0.0) for hit in hits) / max(1, len(hits))
    entity_score = 1.0 if not context.entities else float(entity_report.get("coverage", 0.0) or 0.0)
    contradiction = 1.0 if bool(contradiction_report.get("has_contradiction")) else 0.0
    hard_pass = bool(final_verification.get("allow"))
    confidence = compute_confidence(
        retrieval=retrieval,
        evidence_quality=evidence_quality,
        entailment=verification_ratio,
        entity_coverage=entity_score,
        verification=1.0 if hard_pass else 0.0,
        contradiction=contradiction,
        hard_gate_passed=hard_pass,
    )
    evidence = build_evidence_bundle(
        context.request_id,
        hits,
        entity_coverage=entity_report,
        contradiction_count=1 if contradiction else 0,
    )
    answer = str(enhanced.get("answer") or "")
    status = "abstain" if is_control_message(answer) or str(enhanced.get("status", "")).endswith("ABSTAIN") else "verified"
    checks = list(final_verification.get("claim_checks") or ())
    matrix = list(final_verification.get("evidence_claim_matrix") or enhanced.get("evidence_claim_matrix") or ())
    blocked = int(final_verification.get("blocked_claims", 0) or 0)
    enhanced["request_id"] = context.request_id
    enhanced["contract_version"] = CONTRACT_VERSION
    enhanced["request_context"] = context.to_dict()
    enhanced["evidence_bundle"] = evidence.to_dict()
    enhanced["confidence_breakdown"] = confidence.to_dict()
    enhanced["confidence_calibration"] = {
        "level": confidence.level,
        "calibrated": confidence.final,
        "retrieval": confidence.retrieval,
        "evidence_quality": confidence.evidence_quality,
        "entailment": confidence.entailment,
        "entity_coverage": confidence.entity_coverage,
        "verification": confidence.verification,
        "contradiction": confidence.contradiction,
        "policy": "independent_signals_v3",
    }
    enhanced["answer_envelope"] = {
        "request_id": context.request_id,
        "status": status,
        "control_reason": (enhanced.get("abstention_reasons") or [None])[0] if status == "abstain" else None,
        "allow": bool(final_verification.get("allow")),
        "verification_reason": final_verification.get("reason"),
        "claim_count": len(checks) if checks else len(matrix),
        "blocked_claims": blocked,
        "citation_count": len(enhanced.get("citations") or ()),
        "verification": dict(final_verification),
        "confidence": confidence.to_dict(),
    }
    enhanced["claim_matrix_summary"] = {
        "claim_count": len(matrix),
        "entailed_claims": sum(
            1 for record in matrix if isinstance(record, Mapping) and str(record.get("status", "")) == "ENTAILED"
        ),
        "blocked_claims": blocked,
        "all_entailed": bool(matrix) and blocked == 0 and all(
            str(record.get("status", "")) == "ENTAILED" for record in matrix if isinstance(record, Mapping)
        ),
        "supported_ratio": verification_ratio,
    }
    enhanced["contract_status"] = status
    enhanced["confidence"] = {
        "level": confidence.level,
        "evidence_confidence": confidence.final,
        "retrieval": confidence.retrieval,
        "evidence_quality": confidence.evidence_quality,
        "entailment": confidence.entailment,
        "entity_coverage": confidence.entity_coverage,
        "verification": confidence.verification,
        "contradiction": confidence.contradiction,
    }
    return enhanced


def install() -> None:
    """Compatibility registration only; behavioral contract application is explicit."""
    return None


__all__ = [
    "CONTRACT_VERSION", "RequestContext", "EvidenceSource", "EvidenceBundle",
    "ConfidenceBreakdown", "new_request_id", "build_request_context",
    "build_evidence_bundle", "compute_confidence", "apply_contract", "install",
]
