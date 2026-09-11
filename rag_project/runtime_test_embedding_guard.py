from __future__ import annotations

from typing import Any

_INSTALLED = False


def _explicit_test_mode(system: Any) -> bool:
    settings = getattr(system, "settings", None)
    service = getattr(system, "embedding_service", None)
    if service is None:
        return False
    explicit = bool(getattr(settings, "embedding_test_mode", False))
    model_marker = (
        str(getattr(settings, "embedding_model", "") or "").strip().casefold() == "test"
    )
    return explicit or model_marker


def _deterministic_discover_dimension(service: Any) -> int:
    dimension = int(getattr(service, "dimension", 0) or 0)
    if dimension > 0:
        return dimension
    dimension = len(service._test_embedding("__rag_dimension_probe__"))
    service.dimension = dimension
    return dimension


def _wrap_ingest(original: Any):
    def wrapped(self: Any, pdf_path: Any):
        service = getattr(self, "embedding_service", None)
        if service is None or not _explicit_test_mode(self):
            return original(self, pdf_path)

        service.test_mode = True
        service.provider = "deterministic-test"
        service.last_error = None
        service.dimension = _deterministic_discover_dimension(service)
        self.embedding_startup_error = None

        sentinel = object()
        previous = service.__dict__.get("discover_dimension", sentinel)
        service.discover_dimension = lambda: _deterministic_discover_dimension(service)
        try:
            return original(self, pdf_path)
        finally:
            if previous is sentinel:
                service.__dict__.pop("discover_dimension", None)
            else:
                service.__dict__["discover_dimension"] = previous

    wrapped._runtime_test_embedding_guard = True
    wrapped.__name__ = getattr(original, "__name__", "ingest_file")
    wrapped.__qualname__ = getattr(original, "__qualname__", "ingest_file")
    return wrapped


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.app.rag_system import RAGSystem

    current = getattr(RAGSystem, "ingest_file", None)
    if callable(current) and not getattr(current, "_runtime_test_embedding_guard", False):
        RAGSystem.ingest_file = _wrap_ingest(current)
    _INSTALLED = True


__all__ = ["install"]
