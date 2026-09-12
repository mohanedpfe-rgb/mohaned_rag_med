"""Final post-install contract corrections for routing, evidence scope, and numeric functionality."""
from __future__ import annotations

import re
from typing import Any

from rag_project.intelligence.evidence_guard import verify_claims


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


_GENERIC = {
    "what", "which", "where", "when", "who", "why", "how", "does", "the", "is", "are", "a", "an",
    "of", "for", "to", "and", "or", "in", "on", "with", "from", "about", "using", "only", "indexed",
    "evidence", "state", "stated", "say", "says", "list", "table", "figure", "numeric", "information",
    "question", "explain", "compare", "management", "treatment", "mechanism", "clinical", "implications",
    "including", "qu", "est", "ce", "que", "le", "la", "les", "des", "du", "un", "une", "pour",
    "comment", "quelle", "quel", "quels", "quelles", "ما", "هو", "هي", "من", "في", "عن", "داء",
}
_COMMON_MEDICAL = {
    "diabetes", "diabetes mellitus", "metformin", "metformine", "hba1c", "hypertension", "cancer", "anemia",
    "infection", "syndrome", "symptom", "symptoms", "disease", "condition", "diagnosis", "patient", "therapy",
    "treatment", "mechanism", "type", "controlled", "clinical", "dose", "dosage", "frequency", "table", "figure",
    "diabète", "diabete", "médicament", "medicament", "traitement", "maladie", "symptôme", "symptome",
    "metabolique", "métabolique", "سكري", "السكري", "داء السكري", "دواء", "العلاج", "علاج", "مرض",
}


def _hit_text(hit: Any) -> str:
    metadata = " ".join(str(v) for v in (getattr(hit, "metadata", {}) or {}).values())
    return _norm(f"{getattr(hit, 'text', '')} {metadata}")


def _safe_filter_relevant_hits(hits: list[Any], question: str, route: Any) -> list[Any]:
    if not hits:
        return []
    question_text = _norm(question)
    tokens = set(re.findall(r"[\w%/.-]{3,}", question_text, flags=re.UNICODE))
    terms = {t for t in tokens if t not in _GENERIC}
    for entity in getattr(route, "entities", ()) or ():
        et = _norm(entity)
        if len(et) >= 4 and et not in _GENERIC:
            terms.add(et)
            terms.update(re.findall(r"[\w%/.-]{3,}", et, flags=re.UNICODE))

    hard_terms = {
        term for term in terms
        if "_" in term or any(ch.isdigit() for ch in term) or (len(term) >= 9 and term not in _COMMON_MEDICAL)
    }
    if hard_terms and not any(any(term in _hit_text(hit) for term in hard_terms) for hit in hits):
        return []

    expanded = set(terms)
    aliases = {
        "diabetes": {"diabetes", "diabetes mellitus", "diabète", "diabete", "داء السكري", "السكري", "سكري"},
        "diabète": {"diabetes", "diabetes mellitus", "diabète", "diabete", "داء السكري", "السكري", "سكري"},
        "diabete": {"diabetes", "diabetes mellitus", "diabète", "diabete", "داء السكري", "السكري", "سكري"},
        "metformin": {"metformin", "metformine", "ميتفورمين"},
        "metformine": {"metformin", "metformine", "ميتفورمين"},
        "سكري": {"diabetes", "diabetes mellitus", "diabète", "داء السكري", "السكري"},
        "السكري": {"diabetes", "diabetes mellitus", "diabète", "داء السكري", "السكري", "سكري"},
        "داء": {"diabetes", "diabetes mellitus", "diabète", "السكري", "داء السكري"},
    }
    for term in tuple(expanded):
        expanded.update(aliases.get(term, ()))
    matched = [hit for hit in hits if any(term and term in _hit_text(hit) for term in expanded)]
    if matched:
        matched_ids = {id(hit) for hit in matched}
        return matched + [hit for hit in hits if id(hit) not in matched_ids]
    return hits


_NUMERIC_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b",
    re.I,
)
_UNIT_SCALE = {
    "kg": 1_000_000.0,
    "g": 1_000.0,
    "mg": 1.0,
    "mcg": 0.001,
    "µg": 0.001,
    "l": 1_000.0,
    "ml": 1.0,
    "mmhg": 1.0,
    "mmol/l": 1.0,
    "%": 1.0,
    "iu": 1.0,
    "unit": 1.0,
    "units": 1.0,
}
_UNIT_DIMENSION = {
    "kg": "mass", "g": "mass", "mg": "mass", "mcg": "mass", "µg": "mass",
    "l": "volume", "ml": "volume", "mmhg": "pressure", "mmol/l": "concentration",
    "%": "percent", "iu": "activity", "unit": "activity", "units": "activity",
}


def _canonical_unit(unit: str) -> str:
    return str(unit or "").casefold().replace(" ", "")


def _numeric_value(value: float, unit: str) -> tuple[str, float] | None:
    canonical = _canonical_unit(unit)
    dimension = _UNIT_DIMENSION.get(canonical)
    scale = _UNIT_SCALE.get(canonical)
    if dimension is None or scale is None:
        return None
    return dimension, value * scale


def _numeric_unit_equivalent(value_a: str, value_b: str, tolerance: float = 1e-9) -> bool:
    match_a = _NUMERIC_RE.fullmatch(value_a.strip())
    match_b = _NUMERIC_RE.fullmatch(value_b.strip())
    if not match_a or not match_b:
        return False
    try:
        a = _numeric_value(float(match_a.group("value")), match_a.group("unit"))
        b = _numeric_value(float(match_b.group("value")), match_b.group("unit"))
    except (TypeError, ValueError):
        return False
    if a is None or b is None or a[0] != b[0]:
        return False
    return abs(a[1] - b[1]) <= tolerance * max(1.0, abs(a[1]), abs(b[1]))


def _unit_aware_numeric_verifier(original):
    """Ignore a false raw-string mismatch when semantic evidence proves unit equivalence."""
    def wrapped(self: Any, answer: str, hits: Any, route: Any, compiled: dict[str, Any]) -> dict[str, Any]:
        result = dict(original(self, answer, hits, route, compiled) or {})
        if not result.get("numeric_mismatch") or not answer or not hits:
            return result

        blocks = [str(getattr(hit, "text", "") or "") for hit in hits]
        marker_ids = [f"S{i + 1}" for i in range(len(hits))]
        checks = list(verify_claims(answer, blocks, marker_ids))
        semantic_numeric_mismatch = any(
            getattr(check, "numeric_mismatch", False)
            or getattr(check, "status", "") == "NUMERIC_MISMATCH"
            for check in checks
        )
        if semantic_numeric_mismatch:
            return result

        answer_values = [
            match.group(0)
            for match in _NUMERIC_RE.finditer(answer)
        ]
        evidence_values = [
            match.group(0)
            for block in blocks
            for match in _NUMERIC_RE.finditer(block)
        ]
        for answer_value in answer_values:
            if any(_numeric_unit_equivalent(answer_value, evidence_value) for evidence_value in evidence_values):
                grounding = result.get("grounding") if isinstance(result.get("grounding"), dict) else {}
                final = result.get("final_answer") if isinstance(result.get("final_answer"), dict) else {}
                result["numeric_mismatch"] = False
                result["allow"] = bool(grounding.get("allow")) and bool(final.get("allow", True))
                return result
        return result

    wrapped._post_contract_numeric_guard = True
    wrapped._post_contract_numeric_original = original
    return wrapped


def _filtered_cache_cleanup_wrapper(original):
    """Prevent filtered retrieval results from surviving under the global question key."""
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        result = original(self, question, route, where)
        if where is not None:
            cache = getattr(self, "cache", None)
            delete = getattr(cache, "delete", None)
            if callable(delete):
                delete(question)
        return result

    wrapped._post_contract_filtered_cache_guard = True
    wrapped._post_contract_filtered_cache_original = original
    return wrapped


def install() -> None:
    from rag_project.intelligence import med_evidence_pro
    from rag_project import runtime_deep_contract_fix as deep

    deep._filter_relevant_hits = _safe_filter_relevant_hits

    original_route = med_evidence_pro.QueryRouter.route
    if not getattr(original_route, "_post_contract_route", False):
        def route(self: Any, question: str, context: str, safety: Any):
            base = original_route(self, question, context, safety)
            return deep._canonical_route(question, base)
        route._post_contract_route = True
        med_evidence_pro.QueryRouter.route = route

    original_answer = med_evidence_pro.MedEvidenceProEngine.answer
    if not getattr(original_answer, "_post_contract_answer", False):
        def answer(self: Any, question: str, metadata_filter: dict[str, Any] | None = None):
            result = dict(original_answer(self, question, metadata_filter) or {})
            if str(result.get("status") or "").upper() == "NOT_SUPPORTED":
                result["hits"] = []
                result["citations"] = []
                result["generation_path"] = ""
            return result
        answer._post_contract_answer = True
        med_evidence_pro.MedEvidenceProEngine.answer = answer

    original_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
    if not getattr(original_retrieve, "_post_contract_filtered_cache_guard", False):
        med_evidence_pro.MultiTierRetriever.retrieve = _filtered_cache_cleanup_wrapper(original_retrieve)

    original_verify = med_evidence_pro.ActiveVerifier.verify
    if not getattr(original_verify, "_post_contract_numeric_guard", False):
        med_evidence_pro.ActiveVerifier.verify = _unit_aware_numeric_verifier(original_verify)


__all__ = [
    "install",
    "_safe_filter_relevant_hits",
    "_numeric_unit_equivalent",
    "_unit_aware_numeric_verifier",
    "_filtered_cache_cleanup_wrapper",
]
