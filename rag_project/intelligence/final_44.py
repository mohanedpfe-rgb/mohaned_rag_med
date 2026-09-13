"""Compatibility facade for the retired final_44 contract module."""
from __future__ import annotations

import re
from typing import Any, Sequence

from rag_project.intelligence.god_mode import GOD_MODE_FEATURES


def validate_citations(answer: str, sources: Sequence[Any]):
    text = str(answer or "")
    source_count = len(sources)
    valid: list[int] = []
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= source_count:
            valid.append(number)
            return f"[S{number}]"
        invalid.append(number)
        return "[S?]"

    result = re.sub(r"\[S(\d+)\]", replace, text, flags=re.I)
    return result, {"valid": valid, "invalid": invalid}


def validate_feature_registry() -> dict[str, Any]:
    features = tuple(GOD_MODE_FEATURES)
    errors: list[str] = []
    if len(features) != 44:
        errors.append(f"expected 44 features, got {len(features)}")
    if len(set(features)) != len(features):
        errors.append("duplicate feature names")
    return {
        "count": len(features),
        "expected": 44,
        "ok": not errors,
        "errors": errors,
        "features": list(features),
    }


__all__ = ["validate_citations", "validate_feature_registry"]
