"""Last-pass functionality corrections proven against the current production stack."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

_PERCENT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%")
_NUMERIC_TOKEN_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)(?![A-Za-z0-9_])",
    re.I,
)


def _percent_equivalent(left: str, right: str) -> bool:
    def parse(value: str) -> float | None:
        match = _PERCENT_RE.fullmatch(str(value or "").strip())
        if not match:
            return None
        try:
            return float(match.group(0).rstrip("% "))
        except ValueError:
            return None

    a = parse(left)
    b = parse(right)
    return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _wrap_followup_route(original: Any):
    def wrapped(self: Any, question: str, context: str, safety: Any):
        route = original(self, question, context, safety)
        query = re.sub(r"\s+", " ", str(question or "")).strip().casefold()
        explicit = bool(
            re.search(r"\b(?:what about|how about|it|this|that|they|them)\b", query)
            or re.match(r"^(?:and|also|et|puis|و|ثم)\b", query, flags=re.I | re.UNICODE)
        )
        if bool(getattr(route, "is_follow_up", False)) != explicit:
            return replace(route, is_follow_up=explicit)
        return route

    wrapped._functionality_followup_final_guard = True
    return wrapped


def _wrap_percent_numeric_verifier(original: Any):
    def wrapped(self: Any, answer: str, hits: Any, route: Any, compiled: dict[str, Any]):
        result = dict(original(self, answer, hits, route, compiled) or {})
        if not answer or not hits or not bool(getattr(route, "numeric_sensitivity", False)):
            return result

        answer_percent = [m.group(0) for m in _PERCENT_RE.finditer(str(answer))]
        if not answer_percent:
            return result
        evidence_percent = [m.group(0) for hit in hits for m in _PERCENT_RE.finditer(str(getattr(hit, "text", "") or ""))]
        if not evidence_percent or not all(any(_percent_equivalent(a, e) for e in evidence_percent) for a in answer_percent):
            result["numeric_mismatch"] = True
            result["allow"] = False
            return result

        non_percent_answer = []
        for match in _NUMERIC_TOKEN_RE.finditer(str(answer)):
            if match.group("unit") != "%":
                non_percent_answer.append(match.group(0))
        if not non_percent_answer and bool(result.get("numeric_mismatch")):
            grounding = result.get("grounding") if isinstance(result.get("grounding"), dict) else {}
            final = result.get("final_answer") if isinstance(result.get("final_answer"), dict) else {}
            result["numeric_mismatch"] = False
            result["allow"] = bool(grounding.get("allow")) and bool(final.get("allow", True))
        return result

    wrapped._functionality_percent_numeric_guard = True
    return wrapped


def _wrap_percent_contradictions(original: Any):
    def wrapped(claims: Any):
        result = dict(original(claims) or {})
        kept = []
        for conflict in result.get("conflicts") or []:
            left = [str(value) for value in conflict.get("left") or ()]
            right = [str(value) for value in conflict.get("right") or ()]
            comparable = [(a, b) for a in left for b in right if _PERCENT_RE.fullmatch(a.strip()) and _PERCENT_RE.fullmatch(b.strip())]
            if comparable and all(_percent_equivalent(a, b) for a, b in comparable):
                continue
            kept.append(conflict)
        result["conflicts"] = kept[:8]
        result["has_contradiction"] = bool(kept)
        result["agreement"] = 0.65 if kept else 1.0
        return result

    wrapped._functionality_percent_contradiction_guard = True
    return wrapped


def install() -> None:
    from rag_project.intelligence import med_evidence_pro

    original_route = med_evidence_pro.QueryRouter.route
    if not getattr(original_route, "_functionality_followup_final_guard", False):
        med_evidence_pro.QueryRouter.route = _wrap_followup_route(original_route)

    original_verify = med_evidence_pro.ActiveVerifier.verify
    if not getattr(original_verify, "_functionality_percent_numeric_guard", False):
        med_evidence_pro.ActiveVerifier.verify = _wrap_percent_numeric_verifier(original_verify)

    original_contradiction = med_evidence_pro.EvidenceCompiler._detect_contradiction
    if not getattr(original_contradiction, "_functionality_percent_contradiction_guard", False):
        med_evidence_pro.EvidenceCompiler._detect_contradiction = staticmethod(_wrap_percent_contradictions(original_contradiction))


__all__ = [
    "install",
    "_wrap_followup_route",
    "_wrap_percent_numeric_verifier",
    "_wrap_percent_contradictions",
]
