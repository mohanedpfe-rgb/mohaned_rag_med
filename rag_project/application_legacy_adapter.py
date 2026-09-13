"""Explicit compatibility adapter for the legacy RAG implementation.

The canonical application layer uses this adapter instead of inheriting from the
legacy Streamlit-era service. The adapter is intentionally the only place where
that compatibility dependency is allowed to cross the application boundary.
"""
from __future__ import annotations

from typing import Any

from rag_project.app.production_rag import ProductionRAGSystem


class LegacyProductionRAGAdapter:
    """Delegate the established runtime surface to the legacy implementation."""

    __slots__ = ("_delegate",)

    def __init__(self, settings: Any) -> None:
        object.__setattr__(self, "_delegate", ProductionRAGSystem(settings))

    @property
    def delegate(self) -> ProductionRAGSystem:
        return object.__getattribute__(self, "_delegate")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_delegate":
            object.__setattr__(self, name, value)
            return
        setattr(self.delegate, name, value)

    def ingest_file(self, pdf_path: Any) -> Any:
        return self.delegate.ingest_file(pdf_path)

    def ingest_directory(self, directory: Any = None) -> Any:
        return self.delegate.ingest_directory(directory)

    def health_report(self) -> dict[str, Any]:
        return dict(self.delegate.health_report() or {})

    def cancel_all_ingests(self) -> Any:
        method = getattr(self.delegate, "cancel_all_ingests", None)
        return method() if callable(method) else 0


__all__ = ["LegacyProductionRAGAdapter"]
