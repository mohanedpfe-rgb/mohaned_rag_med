"""Final post-install contract corrections for routing and evidence scope."""
from __future__ import annotations

import re
from typing import Any


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


__all__ = ["install"]
