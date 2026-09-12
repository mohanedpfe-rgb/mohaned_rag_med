"""Deep runtime contract fixes discovered by the high-level production suite.

This module contains narrow, idempotent compatibility patches for three defects:
1. nested runtime-safety boundaries hiding the original infrastructure exception;
2. retrieval cache short-circuiting live outage detection;
3. contradiction detection treating different measurement dimensions as conflicts.

The patches preserve the single production answer authority and fail closed.
"""
from __future__ import annotations

import re
import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False
_TLS = threading.local()

_NUMERIC_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|g|kg|mL|ml|L|mmHg|mmol/L|%|IU|units?)\b",
    re.I,
)
_UNIT_DIMENSION = {
    "mg": "mass", "mcg": "mass", "µg": "mass", "g": "mass", "kg": "mass",
    "ml": "volume", "l": "volume", "mmhg": "pressure", "mmol/l": "concentration",
    "%": "percent", "iu": "activity", "unit": "activity", "units": "activity",
}
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "and", "or",
    "in", "on", "with", "by", "as", "this", "that", "these", "those", "value", "values",
    "approximately", "about", "from", "than", "between", "twice", "daily",
}


def _canonical_unit(unit: str) -> str:
    return str(unit or "").casefold().replace(" ", "")


def _dimension(unit: str) -> str:
    return _UNIT_DIMENSION.get(_canonical_unit(unit), "unknown")


def _claim_context(text: str) -> set[str]:
    cleaned = _NUMERIC_RE.sub(" VALUE ", str(text or "").casefold())
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned, flags=re.UNICODE)
    return {
        token for token in re.findall(r"[\w-]{3,}", cleaned, flags=re.UNICODE)
        if token not in _STOPWORDS
    }


def _numeric_groups(text: str) -> list[tuple[float, str, str]]:
    groups: list[tuple[float, str, str]] = []
    for match in _NUMERIC_RE.finditer(str(text or "")):
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        unit = _canonical_unit(match.group("unit"))
        groups.append((value, unit, _dimension(unit)))
    return groups


def _detect_contradiction(claims: Any) -> dict[str, Any]:
    """Detect only context-linked conflicts in the same physical dimension."""
    rows: list[tuple[list[tuple[float, str, str]], set[str], str]] = []
    for claim in claims or ():
        text = str(getattr(claim, "text", "") or "")
        numbers = _numeric_groups(text)
        if numbers:
            rows.append((numbers, _claim_context(text), text))

    conflicts: list[dict[str, Any]] = []
    for index, (left_numbers, left_context, left_text) in enumerate(rows):
        for right_numbers, right_context, right_text in rows[index + 1:]:
            shared = left_context & right_context
            if len(shared) < 2:
                continue
            for left_value, left_unit, left_dim in left_numbers:
                for right_value, right_unit, right_dim in right_numbers:
                    if left_dim == "unknown" or left_dim != right_dim:
                        continue
                    if left_value == right_value and left_unit == right_unit:
                        continue
                    conflicts.append({
                        "left": [f"{left_value:g} {left_unit}"],
                        "right": [f"{right_value:g} {right_unit}"],
                        "dimension": left_dim,
                        "shared_context": sorted(shared)[:8],
                        "left_claim": left_text,
                        "right_claim": right_text,
                    })
    return {
        "has_contradiction": bool(conflicts),
        "conflicts": conflicts[:8],
        "agreement": 0.65 if conflicts else 1.0,
        "method": "dimension_and_context_aware_numeric_conflict",
    }


def _guarded_runtime_safety(original):
    def wrapped(system: Any, question: str, answer_fn):
        depth = int(getattr(_TLS, "runtime_safety_depth", 0) or 0)
        if depth > 0:
            return answer_fn()
        _TLS.runtime_safety_depth = depth + 1
        try:
            return original(system, question, answer_fn)
        finally:
            _TLS.runtime_safety_depth = depth
    wrapped._deep_contract_guard = True
    wrapped._deep_contract_original = original
    return wrapped


def _cache_observability_wrapper(original):
    """Keep cache benefits while proving live retriever health before answer authority."""
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        cache = getattr(self, "cache", None)
        retriever = getattr(self.system, "retriever", None)

        if retriever is not None and callable(getattr(retriever, "retrieve", None)):
            # The retriever is the live infrastructure authority. A one-hit probe
            # prevents a cached response from hiding a current retrieval outage.
            retriever.retrieve(question, 1, where)

        if cache is not None and where is None:
            cached = cache.get(question)
            if cached:
                restored = cache.restore(cached)
                return restored, {
                    "tier": "CACHE",
                    "cache_hit": True,
                    "early_exit": True,
                    "tier0_confidence": self._confidence(restored, route.entities),
                    "retrieval_latency_ms": 0.2,
                    "candidate_count": len(restored),
                }
        return original(self, question, route, where)
    wrapped._deep_contract_cache_guard = True
    wrapped._deep_contract_original = original
    return wrapped


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        from rag_project.intelligence import med_evidence_pro
        from rag_project.intelligence import runtime_safety
        from rag_project import application
        from rag_project.app import production_rag as production_rag_module

        original_runtime_safety = runtime_safety.execute_with_runtime_safety
        if not getattr(original_runtime_safety, "_deep_contract_guard", False):
            guarded = _guarded_runtime_safety(original_runtime_safety)
            runtime_safety.execute_with_runtime_safety = guarded
            application.execute_with_runtime_safety = guarded
            production_rag_module.execute_with_runtime_safety = guarded

        original_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(original_retrieve, "_deep_contract_cache_guard", False):
            med_evidence_pro.MultiTierRetriever.retrieve = _cache_observability_wrapper(original_retrieve)

        med_evidence_pro.EvidenceCompiler._detect_contradiction = staticmethod(_detect_contradiction)
        _INSTALLED = True


__all__ = ["install"]
