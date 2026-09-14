"""Deprecated compatibility module for invariant helpers.

Invariant enforcement now belongs to the owning storage/ingestion/PDF modules.
This module intentionally performs no runtime monkey-patching.
"""
from __future__ import annotations

from typing import Any


def _safe_chroma_metadata(metadata: Any) -> dict[str, Any]:
    value = dict(metadata or {}) if isinstance(metadata, dict) else {}
    for key, item in list(value.items()):
        if isinstance(item, tuple):
            value[key] = list(item)
        if isinstance(value.get(key), list) and not value[key]:
            value.pop(key, None)
    return value


def _direct_table_extract(page: Any) -> str:
    """Compatibility helper; table extraction remains owned by PDFExtractor."""
    try:
        finder = getattr(page, "find_tables", None)
        if not callable(finder):
            return ""
        result = finder()
        rendered: list[str] = []
        for table in list(getattr(result, "tables", []) or [])[:20]:
            try:
                markdown = str(table.to_markdown() or "").strip()
            except Exception:
                continue
            if markdown:
                rendered.append(markdown)
        return "\n\n".join(rendered)
    except Exception:
        return ""


def install() -> None:
    """Deprecated no-op. Invariants are implemented at their ownership layer."""
    return None


__all__ = ["install", "_safe_chroma_metadata", "_direct_table_extract"]
