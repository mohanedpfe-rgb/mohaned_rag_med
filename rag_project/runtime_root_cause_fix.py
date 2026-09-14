"""Compatibility helpers only.

This module deliberately does not monkey-patch retrieval, answer, or publication
methods. The production runtime owns those contracts directly.
"""
from __future__ import annotations

import re
from typing import Any


def _explicit_markers(question: str) -> set[str]:
    return {value.casefold() for value in re.findall(r"\b(?:DOC|SOURCE|VERSION|MARKER|CHUNK)[_-][A-Za-z0-9_-]+\b", str(question or ""), flags=re.I)}


def _scope_filter(hits: list[Any], where: dict[str, Any] | None) -> list[Any]:
    if not where:
        return list(hits)
    def matches(meta: dict[str, Any], condition: Any) -> bool:
        if not condition:
            return True
        if isinstance(condition, dict) and "$and" in condition:
            return all(matches(meta, item) for item in condition.get("$and") or [])
        if isinstance(condition, dict) and "$or" in condition:
            return any(matches(meta, item) for item in condition.get("$or") or [])
        return all(meta.get(str(key)) == value for key, value in dict(condition).items())
    return [hit for hit in hits if matches(dict(getattr(hit, "metadata", {}) or {}), where)]


def _document_scope_filter(hits: list[Any], question: str) -> list[Any]:
    markers = _explicit_markers(question)
    if not markers:
        return list(hits)
    selected: list[Any] = []
    for hit in hits:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        haystack = " ".join([str(getattr(hit, "text", "") or ""), " ".join(str(v) for v in metadata.values())]).casefold()
        if any(marker in haystack for marker in markers):
            selected.append(hit)
    return selected


def install() -> None:
    """No-op compatibility entrypoint.

    Historical callers may still invoke this installer, but it must never mutate
    production method identities. Retrieval scoping is implemented by the
    canonical MultiTierRetriever implementation and publication/memory semantics
    are handled at their owning boundaries.
    """
    return None


__all__ = ["install", "_explicit_markers", "_scope_filter", "_document_scope_filter"]
