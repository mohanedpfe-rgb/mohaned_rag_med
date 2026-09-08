"""Application composition root for production-facing services."""

from __future__ import annotations

import threading
from typing import Any

from rag_project.configuration.settings import Settings
from rag_project.runtime import install

_FACTORY_LOCK = threading.RLock()


def create_rag_system(settings: Settings | None = None):
    """Construct the production RAG service after installing runtime policies."""
    with _FACTORY_LOCK:
        install()
        from rag_project.app.resilient_rag import ResilientRAGSystem

        return ResilientRAGSystem(settings or Settings.from_env())


def create_default_rag_system():
    """Construct the application from environment-backed settings."""
    return create_rag_system()


def runtime_contract() -> dict[str, Any]:
    """Return stable architecture metadata for health checks and diagnostics."""
    return {
        "composition_root": "rag_project.application.create_rag_system",
        "runtime_policy": "installed_before_service_construction",
        "configuration": "Settings.from_env",
        "service": "ResilientRAGSystem",
    }


__all__ = ["create_rag_system", "create_default_rag_system", "runtime_contract"]
