"""Functionality correction for intent precedence in mixed comparison/numeric queries."""
from __future__ import annotations

from dataclasses import replace
from typing import Any


def _wrap_route(original: Any):
    def wrapped(self: Any, question: str, context: str, safety: Any):
        route = original(self, question, context, safety)
        query = str(question or "").casefold()
        comparison_markers = ("compare", "comparison", "difference", "differences", "versus", " vs ", "between", "différence", "comparaison", "مقارنة", "فرق")
        if route.numeric_sensitivity and any(marker in query for marker in comparison_markers):
            return replace(route, intent="comparison", template_type="comparison")
        return route

    wrapped._functionality_routing_precedence_fix = True
    return wrapped


def install() -> None:
    from rag_project.intelligence.med_evidence_pro import QueryRouter

    original = QueryRouter.route
    if not getattr(original, "_functionality_routing_precedence_fix", False):
        QueryRouter.route = _wrap_route(original)


__all__ = ["_wrap_route", "install"]
