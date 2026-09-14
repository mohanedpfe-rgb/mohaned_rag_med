"""Explicit, finite compatibility adapter for the legacy RAG implementation."""
from __future__ import annotations

from typing import Any

from rag_project.ingestion import versioned_ingestor


class LegacyProductionRAGAdapter:
    """Expose only the infrastructure interface needed by the canonical service.

    The legacy object remains behind this boundary. Unknown attributes are not
    forwarded automatically, preventing accidental authority leakage back into
    the legacy application package.
    """

    __slots__ = ("_delegate", "_local")

    _DELEGATED_ATTRIBUTES = frozenset({
        "settings", "state_store", "vector_store", "retriever", "conversation_memory",
        "logger", "embedding_identity", "embedder", "parser", "cloud_hybrid",
        "startup_quality", "_active_metadata_filter", "_answer_service_corpus_generation",
        "_hash_file", "health_report", "ingest_directory", "ingest_file",
        "cancel_all_ingests", "cancel_ingest", "_new_cancel_flag", "_remove_cancel_flag",
        "clear_pdf_data", "apply_settings_in_place",
    })

    def __init__(self, settings: Any) -> None:
        # Resolve the legacy service at construction time so runtime adapters,
        # diagnostics, and controlled test substitutions all bind to the same
        # canonical module object instead of a stale import-time class.
        from rag_project.app import production_rag
        object.__setattr__(self, "_delegate", production_rag.ProductionRAGSystem(settings))
        object.__setattr__(self, "_local", {})

    @property
    def delegate(self) -> Any:
        return object.__getattribute__(self, "_delegate")

    def __getattr__(self, name: str) -> Any:
        if name in self._DELEGATED_ATTRIBUTES:
            return getattr(self.delegate, name)
        local = object.__getattribute__(self, "_local")
        if name in local:
            return local[name]
        raise AttributeError(f"LegacyProductionRAGAdapter exposes no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_delegate", "_local"}:
            object.__setattr__(self, name, value)
            return
        if name in self._DELEGATED_ATTRIBUTES:
            setattr(self.delegate, name, value)
            return
        object.__getattribute__(self, "_local")[name] = value

    def answer(self, question: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """Route every production answer through the canonical answer service."""
        from rag_project.application_answer_service import answer as canonical_answer
        return canonical_answer(self, question, metadata_filter)

    def ingest_file(self, pdf_path: Any) -> Any:
        return versioned_ingestor.ingest_version_safely(self.delegate, pdf_path)

    def ingest_directory(self, directory: Any = None) -> Any:
        from pathlib import Path
        root = Path(directory or self.settings.incoming_dir)
        return [self.ingest_file(path) for path in sorted(root.glob("*.pdf"))]

    def health_report(self) -> dict[str, Any]:
        return dict(self.delegate.health_report() or {})

    def cancel_all_ingests(self) -> Any:
        method = getattr(self.delegate, "cancel_all_ingests", None)
        return method() if callable(method) else 0

    def cancel_ingest(self, document_id: str) -> bool:
        method = getattr(self.delegate, "cancel_ingest", None)
        return bool(method(document_id)) if callable(method) else False

    def __dir__(self) -> list[str]:
        local = object.__getattribute__(self, "_local")
        return sorted(set(super().__dir__()) | self._DELEGATED_ATTRIBUTES | set(local))


__all__ = ["LegacyProductionRAGAdapter"]
