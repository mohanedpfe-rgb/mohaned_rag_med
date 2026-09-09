"""Application composition root for production-facing services."""

from __future__ import annotations

import threading
from typing import Any

from rag_project.configuration.settings import Settings
from rag_project.quality_gate import run_quality_gate
from rag_project.runtime import install
from rag_project.security import harden_system

_FACTORY_LOCK = threading.RLock()


def _normalize_runtime_settings(settings: Settings | None) -> Settings:
    """Apply non-negotiable local embedding bounds before any client is constructed."""
    resolved = settings or Settings.from_env()
    resolved.embedding_batch_size = max(16, min(int(resolved.embedding_batch_size), 32))
    resolved.embedding_retries = max(1, min(int(resolved.embedding_retries), 3))
    resolved.embedding_timeout_seconds = max(30.0, min(float(resolved.embedding_timeout_seconds), 300.0))
    resolved.max_workers = max(1, min(int(resolved.max_workers), 4))
    resolved.ollama_concurrency = max(1, min(int(resolved.ollama_concurrency), 2))
    return resolved


def create_rag_system(settings: Settings | None = None):
    """Construct the production RAG pipeline and validate its persistent indexes before use."""
    with _FACTORY_LOCK:
        install()
        from rag_project.app.production_rag import ProductionRAGSystem

        resolved_settings = _normalize_runtime_settings(settings)
        system = ProductionRAGSystem(resolved_settings)
        system = harden_system(system)
        try:
            system.startup_quality = run_quality_gate(system, repair_drift=True)
            if not system.startup_quality.get("ready", False):
                system.logger.warning("Runtime quality gate reported a non-ready state: %s", system.startup_quality)
        except Exception as exc:
            system.startup_quality = {
                "ready": False,
                "error": type(exc).__name__,
            }
            system.logger.exception("Runtime quality gate failed")
        return system


def create_default_rag_system():
    """Construct the application from environment-backed settings."""
    return create_rag_system()


def runtime_contract() -> dict[str, Any]:
    return {
        "composition_root": "rag_project.application.create_rag_system",
        "runtime_policy": "infrastructure_installed_before_service_construction",
        "security_policy": "core_guards_installed_at_composition_root",
        "quality_policy": "persistent_index_contract_checked_and_repaired_at_startup",
        "configuration": "Settings.from_env",
        "service": "ProductionRAGSystem",
        "answer_pipeline": "explicit_delegation",
        "answer_monkey_patch": False,
        "medical_safety_gate": True,
    }


__all__ = ["create_rag_system", "create_default_rag_system", "runtime_contract"]
