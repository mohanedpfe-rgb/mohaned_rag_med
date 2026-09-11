from __future__ import annotations

import re
from typing import Any, Sequence

_INSTALLED = False


def _clean_follow_up(question: str, history: Sequence[tuple[str, str]] | None = None) -> str:
    """Authoritative clean public follow-up contract.

    Standalone questions are returned unchanged. Explicit follow-ups are expanded
    with recent conversational context, but the public top-level API never emits
    the legacy ``Follow-up:`` marker.
    """
    cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
    if not cleaned or not history:
        return cleaned

    explicit = bool(
        re.search(r"\b(what about|how about|it|this|that|they|them|those|these)\b", cleaned, re.I)
        or re.match(r"^(and|also|then|et|puis|و|ثم)\b", cleaned, re.I | re.UNICODE)
        or cleaned.startswith(("و", "ثم", "هذا", "هذه", "ذلك", "تلك"))
    )
    if not explicit:
        return cleaned

    recent = list(history)[-3:]
    anchor_question = next(
        (str(q or "").strip() for q, _ in reversed(recent) if str(q or "").strip()),
        "",
    )
    anchor_answer = next(
        (str(a or "").strip() for q, a in reversed(recent) if str(q or "").strip() and str(a or "").strip()),
        "",
    )
    if not anchor_question:
        return cleaned

    terms: list[str] = []
    try:
        from rag_project.intelligence.semantic_reasoning import extract_clinical_entities

        for entity in extract_clinical_entities(anchor_answer):
            value = str(entity.normalized or entity.text or "").strip()
            if value and value.casefold() not in {x.casefold() for x in terms}:
                terms.append(value)
    except Exception:
        pass

    for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]{5,}\b", anchor_answer):
        if token.casefold() not in {x.casefold() for x in terms}:
            terms.append(token)
        if len(terms) >= 4:
            break

    return " ".join(x for x in (anchor_question, " ".join(terms[:4]), cleaned) if x).strip()[:3500]


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.intelligence import top_level_pipeline

    top_level_pipeline.rewrite_follow_up = _clean_follow_up
    _INSTALLED = True
