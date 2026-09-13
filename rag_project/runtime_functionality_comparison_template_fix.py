"""Functionality correction for unsafe side attribution in comparison templates."""
from __future__ import annotations

from typing import Any


def _wrap_comparison_template(original: Any):
    def wrapped(self: Any, compiled: dict[str, Any], route: Any):
        if getattr(route, "template_type", None) != "comparison":
            return original(self, compiled, route)
        claims = compiled.get("claims") or []
        if not claims:
            return None
        return "\n".join(
            f"- Comparison evidence: {claim.text} [S{claim.source_numbers[0]}]"
            for claim in claims[:8]
        )

    wrapped._functionality_comparison_template_fix = True
    return wrapped


def install() -> None:
    from rag_project.intelligence.med_evidence_pro import AnswerCascade

    original = AnswerCascade._template
    if not getattr(original, "_functionality_comparison_template_fix", False):
        AnswerCascade._template = staticmethod(_wrap_comparison_template(original))


__all__ = ["_wrap_comparison_template", "install"]
