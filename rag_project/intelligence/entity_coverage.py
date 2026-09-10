from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Sequence

from rag_project.intelligence.semantic_reasoning import extract_clinical_entities, normalize_medical_term
from rag_project.utils.text_utils import meaningful_tokens

_MEASUREMENT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|ml|mL|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|C|mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)(?=\s|$|[^\w])",
    re.I,
)
_ABBREVIATION = re.compile(r"\b[A-Z]{2,8}(?:[-/][A-Z0-9]{1,8})?\b")
_WORD = re.compile(r"\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'/-]*\b", re.UNICODE)
_STOP = {"WHAT", "THIS", "THAT", "WHICH", "WHERE", "WHEN", "WITH", "FROM", "AND", "THE", "FOR", "DOES", "HOW", "WHY", "BETWEEN", "IS", "ARE", "WAS", "WERE", "CAN", "COULD", "WOULD", "SHOULD", "RELATIONSHIP", "RELATION", "TREATMENT", "ABOUT", "PLEASE"}
_OPEN_SET_MEDICAL_SUFFIXES = (
    "gliflozin", "gliptin", "glutide", "parin", "pril", "sartan", "olol", "azole", "cillin",
    "mycin", "cycline", "vir", "mab", "nib", "tinib", "caine", "statin", "oxetine", "pramine",
    "pam", "lam", "zepam", "barb", "bital", "caine", "phylline", "terol", "lukast", "setron",
)
_OPEN_SET_MEDICAL_TERMS = {"dapagliflozin"}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _canonical(value: str) -> str:
    try:
        return normalize_medical_term(value)
    except Exception:
        return _norm(value)


def _lexical_open_set_terms(text: str) -> list[str]:
    """Find plausible biomedical tokens absent from the closed alias dictionary.

    This is deliberately conservative: drug-name morphology and a tiny explicit
    vocabulary are used instead of treating every ordinary English word as an entity.
    """
    found: list[str] = []
    for match in _WORD.finditer(text or ""):
        token = match.group(0)
        normalized = _norm(token)
        upper = token.upper()
        if not normalized or upper in _STOP or len(normalized) < 5:
            continue
        if normalized in _OPEN_SET_MEDICAL_TERMS or normalized.endswith(_OPEN_SET_MEDICAL_SUFFIXES):
            found.append(_canonical(normalized))
    return found


def extract_query_entities(question: str, planned_entities: Iterable[str] = ()) -> tuple[str, ...]:
    """Build an open-set query entity inventory from the project's existing entity extractor."""
    found: list[str] = []
    try:
        found.extend(_canonical(entity.normalized or entity.text) for entity in extract_clinical_entities(question))
    except Exception:
        pass

    found.extend(_lexical_open_set_terms(question or ""))

    for entity in planned_entities:
        text = _norm(entity)
        if text:
            found.append(_canonical(text))

    for match in _MEASUREMENT.finditer(question or ""):
        found.append(_norm(match.group(0)))
    for match in _ABBREVIATION.finditer(question or ""):
        token = match.group(0)
        if token.upper() not in _STOP:
            found.append(_canonical(token))

    for match in re.finditer(r"\b(?:[A-Za-zÀ-ÿ][\wÀ-ÿ'/-]*\s+){1,3}[A-Za-zÀ-ÿ][\wÀ-ÿ'/-]*\b", question or ""):
        phrase = match.group(0).strip(" ,.;:!?()[]{}")
        if len(meaningful_tokens(phrase)) >= 2:
            found.append(_canonical(phrase))

    unique: list[str] = []
    seen: set[str] = set()
    for value in found:
        value = _norm(value)
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
        if len(unique) >= 32:
            break
    return tuple(unique)


def _entity_overlap(query_entity: str, evidence_entity: str) -> float:
    q = set(meaningful_tokens(query_entity))
    e = set(meaningful_tokens(evidence_entity))
    if not q or not e:
        return 0.0
    if query_entity == evidence_entity:
        return 1.0
    return len(q & e) / max(1, len(q))


def score_entity_coverage(question: str, evidence: Sequence[Any], planned_entities: Iterable[str] = ()) -> dict[str, Any]:
    """Measure whether requested entities are actually represented in retrieved evidence."""
    query_entities = extract_query_entities(question, planned_entities)
    evidence_entities: list[tuple[str, str]] = []
    planned_norms = {_norm(x) for x in planned_entities if _norm(x)}
    for index, hit in enumerate(evidence):
        text = str(getattr(hit, "text", "") or "")
        source_id = f"S{index + 1}"
        try:
            extracted = extract_clinical_entities(text)
        except Exception:
            extracted = ()
        for entity in extracted:
            normalized = _canonical(entity.normalized or entity.text)
            if normalized:
                evidence_entities.append((normalized, source_id))
        for lexical in _lexical_open_set_terms(text):
            evidence_entities.append((lexical, source_id))
        for planned in planned_norms:
            if re.search(rf"(?<![\w-]){re.escape(planned)}(?![\w-])", text, re.I):
                evidence_entities.append((planned, source_id))
        for match in _MEASUREMENT.finditer(text):
            evidence_entities.append((_norm(match.group(0)), source_id))
        for match in _ABBREVIATION.finditer(text):
            token = match.group(0)
            if token.upper() not in _STOP:
                evidence_entities.append((_canonical(token), source_id))

    evidence_norms = [name for name, _ in evidence_entities]
    source_counts = Counter(source for _, source in evidence_entities)
    covered: list[str] = []
    missing: list[str] = []
    matches: list[dict[str, Any]] = []
    per_entity: list[dict[str, Any]] = []

    for entity in query_entities:
        best = max((_entity_overlap(entity, candidate) for candidate in evidence_norms), default=0.0)
        candidate_sources = tuple(dict.fromkeys(source for candidate, source in evidence_entities if _entity_overlap(entity, candidate) >= max(0.72, best - 0.05)))
        status = "covered" if best >= 0.72 else "partial" if best >= 0.45 else "missing"
        row = {"entity": entity, "status": status, "match_score": round(best, 3), "sources": candidate_sources}
        per_entity.append(row)
        if status == "covered":
            covered.append(entity)
        elif status == "missing":
            missing.append(entity)
        if candidate_sources:
            matches.append({"entity": entity, "evidence_sources": candidate_sources, "match_score": round(best, 3)})

    coverage = len(covered) / max(1, len(query_entities))
    partial_coverage = (len(covered) + 0.5 * sum(row["status"] == "partial" for row in per_entity)) / max(1, len(query_entities))
    return {
        "query_entities": list(query_entities),
        "entity_count": len(query_entities),
        "covered": covered,
        "missing": missing,
        "partial": [row["entity"] for row in per_entity if row["status"] == "partial"],
        "coverage": round(coverage, 3),
        "partial_coverage": round(partial_coverage, 3),
        "per_entity": per_entity,
        "evidence_entity_count": len(evidence_entities),
        "evidence_entities": list(dict.fromkeys(evidence_norms))[:64],
        "source_entity_counts": dict(source_counts),
        "matches": matches,
    }


__all__ = ["extract_query_entities", "score_entity_coverage"]
