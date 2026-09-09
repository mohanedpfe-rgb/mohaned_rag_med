"""Application composition root for production-facing services."""

from __future__ import annotations

import threading
from typing import Any

from rag_project.configuration.settings import Settings
from rag_project.runtime import install
from rag_project.security import harden_system

_FACTORY_LOCK = threading.RLock()


def create_rag_system(settings: Settings | None = None):
    """Construct the production RAG pipeline and install security controls before use."""
    with _FACTORY_LOCK:
        install()
        from rag_project.app.production_rag import ProductionRAGSystem

        system = ProductionRAGSystem(settings or Settings.from_env())
        return harden_system(system)


def create_default_rag_system():
    """Construct the application from environment-backed settings."""
    return create_rag_system()


def runtime_contract() -> dict[str, Any]:
    return {
        "composition_root": "rag_project.application.create_rag_system",
        "runtime_policy": "infrastructure_installed_before_service_construction",
        "security_policy": "core_guards_installed_at_composition_root",
        "configuration": "Settings.from_env",
        "service": "ProductionRAGSystem",
        "answer_pipeline": "explicit_delegation",
        "answer_monkey_patch": False,
        "medical_safety_gate": True,
    }


__all__ = ["create_rag_system", "create_default_rag_system", "runtime_contract"]
