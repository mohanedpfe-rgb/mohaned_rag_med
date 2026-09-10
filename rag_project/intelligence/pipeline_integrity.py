"""Production integrity fixes for query contamination and answer-gate failures.

This module is intentionally small and policy-focused.  It is installed from the
application composition root so the existing ingestion, vector store and retrieval
implementations remain untouched while their contracts are made safer.

Core invariants:
- internal diagnostics never become searchable query text;
- only validated clinical concepts are counted as query entities;
- simple factual questions cannot become hard/multi-hop because of metadata labels;
- operational abstention messages are never verified as medical claims.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Sequence

from rag_project.utils.text_utils import meaningful_tokens
from rag_project.intelligence.semantic_reasoning import (
    extract_clinical_entities,
    normalize_medical_term,
)


# These strings are internal protocol labels, not user concepts.  They must never
# be allowed to participate in retrieval, entity extraction, or claim verification.
_INTERNAL_LABELS = {
    "follow-up",
    "follow up",
    "relevant entities",
    "query entities",
    "planned entities",
    "intent",
    "planner",
    "planner confidence",
    "conversation context",
    "semantic understanding",
    "query analysis",
    "retrieval state",
    "adaptive retrieval",
    "evidence alignment",
}

_MEASUREMENT = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|kg|ml|mL|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|C|"
    r"mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)"
    r"(?=\s|$|[^\w])",
    re.I,
)
_ABBREVIATION = re.compile(r"\b[A-Z]{2,8}(?:[-/][A-Z0-9]{1,8})?\b")
_WORD = re.compile(r"\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'/-]*\b", re.UNICODE)
_DRUG_SUFFIXES = (
    "gliflozin", "gliptin", "glutide", "parin", "pril", "sartan", "olol",
    "azole", "cillin", "mycin", "cycline", "vir", "mab", "nib", "tinib",
    "caine", "statin", "oxetine", "pramine", "zepam", "barb", "bital",
    "phylline", "terol", "lukast", "setron", "formin",
)
_CONDITION_SUFFIXES = (
    "itis", "osis", "emia", "pathy", "carcinoma", "oma", "algia", "penia",
    "iasis", "megaly", "cytosis", "trophy", "sclerosis", "stenosis", "ectasia",
)
_STOPWORDS = {
    "what", "what's", "this", "that", "which", "where", "when", "with", "from",
    "and", "the", "for", "does", "how", "why", "are", "is", "was", "were", "can",
    "could", "would", "should", "about", "please", "main", "findings", "finding",
    "relevant", "entities", "follow", "up", "question", "questions", "reported",
    "reports", "document", "documents", "literature", "results", "result",
}

_ABSTENTION_PREFIXES = (
    "i could not verify",
    "i couldn't verify",
    "i could not safely",
    "i cannot safely",
    "i can't safely",
    "the evidence was retrieved, but",
    "the indexed evidence was insufficient",
    "the indexed evidence does not directly support",
    "the retrieved evidence is insufficient",
    "the retrieved evidence is only tangentially related",
    "the generated answer did not meet the evidence-support threshold",
    "the language model is currently unavailable",
    "the language model is unavailable",
    "i could not find sufficient evidence",
    "i couldn't find sufficient evidence",
)



def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _canonical(value: str) -> str:
    try:
        return _normalize(normalize_medical_term(value))
    except Exception:
        return _normalize(value)


def _contains_internal_label(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    return any(label in normalized for label in _INTERNAL_LABELS)


def _open_set_medical_terms(text: str) -> list[str]:
    """Find plausible biomedical tokens without treating generic phrases as entities."""
    found: list[str] = []
    for match in _WORD.finditer(text or ""):
        token = match.group(0)
        normalized = _normalize(token)
        if len(normalized) < 5 or normalized in _STOPWORDS:
            continue
        if normalized.endswith(_DRUG_SUFFIXES) or normalized.endswith(_CONDITION_SUFFIXES):
            canonical = _canonical(normalized)
            if canonical:
                found.append(canonical)
    return found


def safe_extract_query_entities(question: str, planned_entities: Iterable[str] = ()) -> tuple[str, ...]:
    """Return only validated medical concepts, measurements and abbreviations.

    Unlike the previous implementation, this function never turns arbitrary
    two-to-four-word noun phrases (e.g. ``relevant entities``) into entities.
    """
    found: list[str] = []
    deterministic: set[str] = set()

    try:
        for entity in extract_clinical_entities(question or ""):
            raw = entity.normalized or entity.text
            canonical = _canonical(raw)
            if canonical and canonical not in deterministic:
                deterministic.add(canonical)
                found.append(canonical)
    except Exception:
        pass

    # Open-set terms are accepted only at token level and only for strongly
    # medical-looking suffixes.  Generic language is deliberately excluded.
    for term in _open_set_medical_terms(question or ""):
        if term not in found:
            deterministic.add(term)
            found.append(term)

    for match in _MEASUREMENT.finditer(question or ""):
        value = _normalize(match.group(0))
        if value not in found:
            found.append(value)

    for match in _ABBREVIATION.finditer(question or ""):
        token = match.group(0)
        normalized = _normalize(token)
        if normalized in _STOPWORDS or len(normalized) < 2:
            continue
        canonical = _canonical(token)
        # Abbreviations are accepted when the deterministic normalizer knows the
        # concept or when the exact abbreviation is present in the query as a
        # high-signal token (e.g. DKA, CKD, HbA1c).
        if canonical in deterministic or len(token) >= 3:
            for value in (canonical, normalized):
                if value and value not in found:
                    found.append(value)

    planned_allowed = deterministic
    for item in planned_entities:
        raw = _normalize(item)
        if not raw or _contains_internal_label(raw):
            continue
        canonical = _canonical(raw)
        if canonical in planned_allowed or canonical in deterministic:
            if canonical not in found:
                found.append(canonical)

    # Final defense: never expose protocol labels, generic stop words, or phrases
    # that came only from planner prose.
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in found:
        normalized = _normalize(item)
        if (
            not normalized
            or normalized in seen
            or normalized in _STOPWORDS
            or _contains_internal_label(normalized)
        ):
            continue
        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) >= 32:
            break
    return tuple(cleaned)


def _entity_overlap(query_entity: str, evidence_entity: str) -> float:
    q = set(meaningful_tokens(query_entity))
    e = set(meaningful_tokens(evidence_entity))
    if not q or not e:
        return 0.0
    if query_entity == evidence_entity:
        return 1.0
    return len(q & e) / max(1, len(q))


def _evidence_entities(text: str) -> list[str]:
    found: list[str] = []
    try:
        found.extend(_canonical(entity.normalized or entity.text) for entity in extract_clinical_entities(text or ""))
    except Exception:
        pass
    found.extend(_open_set_medical_terms(text or ""))
    for match in _MEASUREMENT.finditer(text or ""):
        found.append(_normalize(match.group(0)))
    for match in _ABBREVIATION.finditer(text or ""):
        token = match.group(0)
        canonical = _canonical(token)
        if canonical not in _STOPWORDS:
            found.extend((canonical, _normalize(token)))
    return [item for item in found if item]


def safe_score_entity_coverage(
    question: str,
    evidence: Sequence[Any],
    planned_entities: Iterable[str] = (),
) -> dict[str, Any]:
    """Score canonical medical entity coverage without generic phrase extraction."""
    query_entities = safe_extract_query_entities(question, planned_entities)
    evidence_entities: list[tuple[str, str]] = []
    for index, hit in enumerate(evidence):
        source_id = f"S{index + 1}"
        text = str(getattr(hit, "text", "") or "")
        for entity in _evidence_entities(text):
            evidence_entities.append((entity, source_id))

    evidence_norms = [name for name, _ in evidence_entities]
    source_counts = Counter(source for _, source in evidence_entities)
    covered: list[str] = []
    missing: list[str] = []
    partial: list[str] = []
    per_entity: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []

    for entity in query_entities:
        best = max((_entity_overlap(entity, candidate) for candidate in evidence_norms), default=0.0)
        candidate_sources = tuple(
            dict.fromkeys(
                source
                for candidate, source in evidence_entities
                if _entity_overlap(entity, candidate) >= max(0.72, best - 0.05)
            )
        )
        status = "covered" if best >= 0.72 else "partial" if best >= 0.45 else "missing"
        row = {
            "entity": entity,
            "status": status,
            "match_score": round(best, 3),
            "sources": candidate_sources,
        }
        per_entity.append(row)
        if status == "covered":
            covered.append(entity)
        elif status == "partial":
            partial.append(entity)
        else:
            missing.append(entity)
        if candidate_sources:
            matches.append(
                {
                    "entity": entity,
                    "evidence_sources": candidate_sources,
                    "match_score": round(best, 3),
                }
            )

    denominator = max(1, len(query_entities))
    coverage = len(covered) / denominator
    partial_coverage = (len(covered) + 0.5 * len(partial)) / denominator
    return {
        "query_entities": list(query_entities),
        "entity_count": len(query_entities),
        "covered": covered,
        "missing": missing,
        "partial": partial,
        "coverage": round(coverage, 3),
        "partial_coverage": round(partial_coverage, 3),
        "per_entity": per_entity,
        "evidence_entity_count": len(evidence_entities),
        "evidence_entities": list(dict.fromkeys(evidence_norms))[:64],
        "source_entity_counts": dict(source_counts),
        "matches": matches,
    }


def _looks_like_followup(question: str, history: Sequence[tuple[str, str]]) -> bool:
    cleaned = _normalize(question)
    if not cleaned or not history:
        return False
    if len(meaningful_tokens(cleaned)) <= 8:
        return True
    return bool(
        re.search(
            r"\b(it|this|that|they|them|those|these|what about|how about|the latter|the former)\b",
            cleaned,
            re.I,
        )
        or re.search(r"^(and|also|then|et|puis|و|ثم)\b", cleaned, re.I | re.UNICODE)
    )


def safe_rewrite_follow_up(
    question: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Resolve actual follow-ups without injecting protocol labels into the query."""
    cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
    if not cleaned or not history or not _looks_like_followup(cleaned, history):
        return cleaned

    recent = list(history[-3:])
    previous_questions = [q.strip() for q, _ in recent if str(q or "").strip()]
    previous_answers = [a.strip() for _, a in recent if str(a or "").strip()]
    anchor_question = previous_questions[-1] if previous_questions else ""
    if not anchor_question:
        return cleaned

    # Prefer deterministic concept anchors from the preceding turn.  They are
    # search terms, not diagnostic annotations.
    anchors: list[str] = []
    for text in (anchor_question, previous_answers[-1] if previous_answers else ""):
        try:
            anchors.extend(entity.normalized for entity in extract_clinical_entities(text)[:8] if entity.normalized)
        except Exception:
            continue

    unique_anchors = list(dict.fromkeys(_normalize(item) for item in anchors if item))
    if unique_anchors:
        # Keep the current utterance as natural language and add only canonical
        # medical anchors that actually occurred in the previous turn.
        return re.sub(r"\s+", " ", f"{cleaned} {' '.join(unique_anchors)}").strip()[:3500]

    # No entity anchor is available.  Preserve the previous question as search
    # context without adding protocol labels such as ``Follow-up:``.
    return re.sub(r"\s+", " ", f"{anchor_question} {cleaned}").strip()[:3500]


def is_control_message(text: str) -> bool:
    """Return True for operational abstention/error messages, not medical answers."""
    normalized = _normalize(text).lstrip("-•* ")
    if not normalized:
        return True
    # Source-only fallbacks contain [S#] markers and are still medical evidence;
    # do not classify them as control messages solely because they mention evidence.
    if re.search(r"\[s\d+\]", normalized, re.I) and not any(normalized.startswith(prefix) for prefix in _ABSTENTION_PREFIXES):
        return False
    return any(normalized.startswith(prefix) for prefix in _ABSTENTION_PREFIXES)


def safe_verify_final_answer(
    answer: str,
    hits: Sequence[Any],
    *,
    require_entailment: bool = False,
) -> dict[str, Any]:
    """Verify actual answer claims while treating controlled abstentions as states."""
    if is_control_message(answer):
        return {
            "checked": False,
            "allow": False,
            "reason": "abstention_not_claim",
            "claim_count": 0,
            "blocked_claims": 0,
            "supported_ratio": 0.0,
            "matrix_claim_count": 0,
            "matrix_all_entailed": False,
            "claim_checks": [],
            "evidence_claim_matrix": [],
        }

    # Import lazily to avoid an import cycle through the intelligence package.
    from rag_project.intelligence.evidence_guard import verify_claims
    from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix

    evidence = [str(getattr(hit, "text", "") or "") for hit in hits]
    source_ids = [f"S{i + 1}" for i in range(len(evidence))]
    checks = verify_claims(answer, evidence, source_ids) if answer.strip() and evidence else []
    matrix = build_claim_evidence_matrix(
        [check.claim for check in checks], hits, source_ids
    ) if checks and hits else ()
    blocked_checks = [
        check for check in checks
        if check.status in {"UNSUPPORTED", "WEAK", "NUMERIC_MISMATCH", "CONTRADICTED"}
        or check.contradiction
    ]
    blocked_matrix = [record for record in matrix if record.status != "ENTAILED"]
    supported = sum(
        check.status in {"SUPPORTED", "PARTIAL"} and not check.contradiction
        for check in checks
    )
    support_ratio = supported / max(1, len(checks))
    matrix_strong = bool(matrix) and not blocked_matrix
    allow = (
        bool(checks)
        and not blocked_checks
        and support_ratio >= 0.60
        and (matrix_strong if require_entailment else True)
    )
    if not checks:
        reason = "no_verifiable_claims"
    elif blocked_checks:
        reason = "blocked_claims"
    elif support_ratio < 0.60:
        reason = "support_ratio_below_threshold"
    elif require_entailment and not matrix_strong:
        reason = "final_matrix_not_fully_entailed"
    else:
        reason = "verified"
    return {
        "checked": bool(checks),
        "allow": allow,
        "reason": reason,
        "claim_count": len(checks),
        "blocked_claims": len(blocked_checks),
        "supported_ratio": round(support_ratio, 4),
        "matrix_claim_count": len(matrix),
        "matrix_all_entailed": matrix_strong,
        "claim_checks": [check.to_dict() for check in checks],
        "evidence_claim_matrix": [record.to_dict() for record in matrix],
    }


def install() -> None:
    """Install integrity policies exactly once in the canonical application process."""
    from rag_project.intelligence import top_level_pipeline
    from rag_project.intelligence import entity_coverage
    from rag_project.intelligence import final_answer_contract
    from rag_project.intelligence import god_mode_100

    if not getattr(top_level_pipeline, "_production_integrity_rewrite_installed", False):
        top_level_pipeline.rewrite_follow_up = safe_rewrite_follow_up
        top_level_pipeline._production_integrity_rewrite_installed = True

    if not getattr(entity_coverage, "_production_integrity_entities_installed", False):
        entity_coverage.extract_query_entities = safe_extract_query_entities
        entity_coverage.score_entity_coverage = safe_score_entity_coverage
        entity_coverage._production_integrity_entities_installed = True

    # god_mode_100 imported these symbols directly, so patch its local references
    # as well as the source modules.
    god_mode_100.score_entity_coverage = safe_score_entity_coverage
    god_mode_100.verify_final_answer = safe_verify_final_answer

    # final_answer_contract is part of the documented runtime contract and is used
    # by callers outside god_mode_100 as well.
    final_answer_contract.verify_final_answer = safe_verify_final_answer


__all__ = [
    "safe_extract_query_entities",
    "safe_score_entity_coverage",
    "safe_rewrite_follow_up",
    "safe_verify_final_answer",
    "is_control_message",
    "install",
]
