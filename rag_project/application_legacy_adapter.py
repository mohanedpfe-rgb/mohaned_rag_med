"""Explicit compatibility adapter for the legacy RAG implementation.

The canonical application layer uses this adapter instead of inheriting from the
legacy Streamlit-era service. The adapter is intentionally the only place where
that compatibility dependency is allowed to cross the application boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_project.app.production_rag import ProductionRAGSystem
from rag_project.ingestion import versioned_ingestor


class LegacyProductionRAGAdapter:
    """Delegate infrastructure to legacy storage while keeping production boundaries canonical."""

    __slots__ = ("_delegate",)

    def __init__(self, settings: Any) -> None:
        candidate = __import__(
            "rag_project.app.production_rag", fromlist=["ProductionRAGSystem"]
        ).ProductionRAGSystem
        service = ProductionRAGSystem
        if candidate is not ProductionRAGSystem and getattr(candidate, "__module__", "") != "rag_project.application":
            service = candidate
        object.__setattr__(self, "_delegate", service(settings))

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

    def answer(self, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """Route every production answer through the canonical MedEvidence service."""
        from rag_project.application_answer_service import answer as canonical_answer
        return canonical_answer(self, question, metadata_filter)

    def ingest_file(self, pdf_path: Any) -> Any:
        return versioned_ingestor.ingest_version_safely(self.delegate, pdf_path)

    def ingest_directory(self, directory: Any = None) -> list[dict[str, Any]]:
        source = Path(directory) if directory is not None else Path(self.delegate.settings.incoming_dir)
        source.mkdir(parents=True, exist_ok=True)
        return [dict(self.ingest_file(path) or {}) for path in sorted(source.glob("*.pdf"))]

    def health_report(self) -> dict[str, Any]:
        return dict(self.delegate.health_report() or {})

    def cancel_all_ingests(self) -> Any:
        method = getattr(self.delegate, "cancel_all_ingests", None)
        return method() if callable(method) else 0


__all__ = ["LegacyProductionRAGAdapter"]