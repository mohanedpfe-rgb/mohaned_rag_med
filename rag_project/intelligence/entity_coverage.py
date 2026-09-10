from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Sequence

from rag_project.intelligence.semantic_reasoning import extract_clinical_entities, normalize_medical_term
from rag_project.utils.text_utils import meaningful_tokens

_MEASUREMENT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|ml|mL|L|mmHg|mmol/L|mol/L|%|IU|units?|bpm|°C|C|mEq/L|mEq|mOsm/L|ng/mL|pg/mL|U/L|kPa)\b",
    re.I,
)
_ABBREVIATION = re.compile(r"\b[A-Z]{2,8}(?:[-/][A-Z0-9]{1,8})?\b")
_STOP = {"WHAT", "THIS", "THAT", "WHICH", "WHERE", "WHEN", "WITH", "FROM", "AND", "THE", "FOR", "DOES", "HOW", "WHY"}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _canonical(value: str) -> str:
    try:
        return normalize_medical_term(value)
    except Exception:
        return _norm(value)


def extract_query_entities(question: str, planned_entities: Iterable[str] = ()) -> tuple[str, ...]:
    """Build an open-set query entity inventory from the project's existing entity extractor."""
    found: list[str] = []
    try:
        found.extend(_canonical(entity.normalized or entity.text) for entity in extract_clinical_entities(question))
    except Exception:
        pass

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

    # Preserve meaningful capitalized / multiword biomedical phrases even when they
    # are absent from the closed alias dictionary. This is intentionally lexical:
    # it never invents a medical concept.
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
    for index, hit in enumerate(evidence):
        text = str(getattr(hit, "text", "") or "")
        try:
            extracted = extract_clinical_entities(text)
        except Exception:
            extracted = ()
        for entity in extracted:
            normalized = _canonical(entity.normalized or entity.text)
            if normalized:
                evidence_entities.append((normalized, f"S{index + 1}"))
        for match in _MEASUREMENT.finditer(text):
            evidence_entities.append((_norm(match.group(0)), f"S{index + 1}"))
        for match in _ABBREVIATION.finditer(text):
            token = match.group(0)
            if token.upper() not in _STOP:
                evidence_entities.append((_canonical(token), f"S{index + 1}"))

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
