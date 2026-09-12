"""Final functionality-only repairs for deterministic answer behavior."""
from __future__ import annotations

import re
import threading
from dataclasses import replace
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False
_TLS = threading.local()


def _wrap_retrieval_cache_fallthrough(original):
    """A cache hit that becomes irrelevant after scope filtering must not become a false abstention."""
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        hits, state = original(self, question, route, where)
        state = dict(state or {})
        if where is None and str(state.get("tier", "")).upper() == "CACHE" and not hits:
            cache = getattr(self, "cache", None)
            delete = getattr(cache, "delete", None)
            if callable(delete):
                delete(question)
            return original(self, question, route, where)
        return hits, state
    wrapped._functionality_cache_fallthrough = True
    return wrapped


def _wrap_retrieval_skip_health_probe(original):
    """Do not let the deep runtime's unused top_k=1 health probe break the real retrieval path."""
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        retriever = getattr(getattr(self, "system", None), "retriever", None)
        method = getattr(retriever, "retrieve", None)
        if not callable(method):
            return original(self, question, route, where)

        called = {"probe": False}
        original_method = method

        def proxy(query: Any, top_k: Any = 8, filter_where: Any = None, *args: Any, **kwargs: Any):
            if (
                not called["probe"]
                and query == question
                and top_k == 1
                and filter_where == where
                and not args
                and not kwargs
            ):
                called["probe"] = True
                return []
            return original_method(query, top_k, filter_where, *args, **kwargs)

        try:
            retriever.retrieve = proxy
            return original(self, question, route, where)
        finally:
            retriever.retrieve = original_method
    wrapped._functionality_probe_guard = True
    return wrapped


def _citation_complete_without_shared_state(original):
    """Validate citations against evidence actually supplied to the current generation."""
    def wrapped(answer: str, hit_count: int) -> bool:
        expected = getattr(_TLS, "citation_limit", None)
        return original(answer, int(expected if expected is not None else hit_count))
    wrapped._functionality_citation_guard = True
    return wrapped


def _wrap_generate(original):
    def wrapped(self: Any, question: str, route: Any, compiled: dict[str, Any]):
        claims = list(compiled.get("claims") or [])
        source_numbers = [
            int(number)
            for claim in claims
            for number in (getattr(claim, "source_numbers", ()) or ())
            if isinstance(number, int) or str(number).isdigit()
        ]
        previous = getattr(_TLS, "citation_limit", None)
        _TLS.citation_limit = max(source_numbers, default=0)
        try:
            return original(self, question, route, compiled)
        finally:
            if previous is None:
                try:
                    del _TLS.citation_limit
                except AttributeError:
                    pass
            else:
                _TLS.citation_limit = previous
    wrapped._functionality_generate_guard = True
    return wrapped


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


def _numeric_unit_equivalent(left: str, right: str) -> bool:
    match_left = _NUMERIC_RE.fullmatch(str(left or "").strip())
    match_right = _NUMERIC_RE.fullmatch(str(right or "").strip())
    if not match_left or not match_right:
        return False
    unit_left = str(match_left.group("unit")).casefold().replace(" ", "")
    unit_right = str(match_right.group("unit")).casefold().replace(" ", "")
    if _UNIT_DIMENSION.get(unit_left) != _UNIT_DIMENSION.get(unit_right):
        return False
    scale_left = _UNIT_SCALE.get(unit_left)
    scale_right = _UNIT_SCALE.get(unit_right)
    if scale_left is None or scale_right is None:
        return False
    try:
        value_left = float(match_left.group("value")) * scale_left
        value_right = float(match_right.group("value")) * scale_right
    except (TypeError, ValueError):
        return False
    return abs(value_left - value_right) <= 1e-9 * max(1.0, abs(value_left), abs(value_right))


def _filter_false_numeric_contradictions(original):
    """Keep genuine numeric conflicts but remove conflicts that are only unit changes."""
    def wrapped(claims: Any):
        result = dict(original(claims) or {})
        conflicts = []
        for conflict in result.get("conflicts") or []:
            left = [str(value) for value in conflict.get("left") or ()]
            right = [str(value) for value in conflict.get("right") or ()]
            comparable = [
                (a, b)
                for a in left
                for b in right
                if _NUMERIC_RE.fullmatch(a.strip()) and _NUMERIC_RE.fullmatch(b.strip())
            ]
            if comparable and all(_numeric_unit_equivalent(a, b) for a, b in comparable):
                continue
            conflicts.append(conflict)
        result["conflicts"] = conflicts[:8]
        result["has_contradiction"] = bool(conflicts)
        result["agreement"] = 0.65 if conflicts else 1.0
        return result
    wrapped._functionality_unit_contradiction_guard = True
    return wrapped


def _functionality_sentences(text: Any) -> list[str]:
    """Keep valid short evidence such as 'DKA.', 'No.', or 'Yes.' instead of dropping it."""
    out: list[str] = []
    for part in re.split(r"(?<=[.!?؟])\s+|\n+", str(text or "")):
        part = re.sub(r"^[-*•\s]+", "", re.sub(r"\s+", " ", part).strip())
        if len(part) >= 2:
            out.append(part)
    return out


def _wrap_route(original):
    """Do not classify normal compound questions containing 'and' as conversational follow-ups."""
    def wrapped(self: Any, question: str, context: str, safety: Any):
        route = original(self, question, context, safety)
        q = re.sub(r"\s+", " ", str(question or "")).strip().casefold()
        explicit = bool(
            re.search(r"\b(?:what about|how about|it|this|that|they|them|also)\b", q)
            or re.match(r"^(?:and|et|puis|و|ثم)\b", q, flags=re.I | re.UNICODE)
        )
        if bool(getattr(route, "is_follow_up", False)) != explicit:
            return replace(route, is_follow_up=explicit)
        return route
    wrapped._functionality_followup_guard = True
    return wrapped


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.intelligence import med_evidence_pro

        original_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(original_retrieve, "_functionality_probe_guard", False):
            original_retrieve = _wrap_retrieval_skip_health_probe(original_retrieve)
            med_evidence_pro.MultiTierRetriever.retrieve = original_retrieve

        current_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(current_retrieve, "_functionality_cache_fallthrough", False):
            med_evidence_pro.MultiTierRetriever.retrieve = _wrap_retrieval_cache_fallthrough(current_retrieve)

        original_citation = med_evidence_pro.AnswerCascade._citation_complete
        if not getattr(original_citation, "_functionality_citation_guard", False):
            med_evidence_pro.AnswerCascade._citation_complete = staticmethod(_citation_complete_without_shared_state(original_citation))

        original_generate = med_evidence_pro.AnswerCascade.generate
        if not getattr(original_generate, "_functionality_generate_guard", False):
            med_evidence_pro.AnswerCascade.generate = _wrap_generate(original_generate)

        original_route = med_evidence_pro.QueryRouter.route
        if not getattr(original_route, "_functionality_followup_guard", False):
            med_evidence_pro.QueryRouter.route = _wrap_route(original_route)

        original_sentences = med_evidence_pro._sentences
        if not getattr(original_sentences, "_functionality_short_sentence_guard", False):
            med_evidence_pro._sentences = _functionality_sentences
            med_evidence_pro._sentences._functionality_short_sentence_guard = True

        original_contradiction = med_evidence_pro.EvidenceCompiler._detect_contradiction
        if not getattr(original_contradiction, "_functionality_unit_contradiction_guard", False):
            guarded = _filter_false_numeric_contradictions(original_contradiction)
            med_evidence_pro.EvidenceCompiler._detect_contradiction = staticmethod(guarded)

        _INSTALLED = True


__all__ = [
    "install",
    "_wrap_retrieval_cache_fallthrough",
    "_wrap_retrieval_skip_health_probe",
    "_citation_complete_without_shared_state",
    "_wrap_generate",
    "_wrap_route",
    "_functionality_sentences",
    "_filter_false_numeric_contradictions",
]
