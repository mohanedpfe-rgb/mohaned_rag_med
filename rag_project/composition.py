"""Production composition boundary.

This module owns runtime configuration normalization and installer ordering.
It deliberately contains no Streamlit UI code so the production composition
sequence can be reused and tested independently of the presentation layer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag_project.canonical_runtime import install as install_canonical_runtime
from rag_project.ingestion.ingestion_contract import install as install_ingestion_contract
from rag_project.intelligence.pipeline_integrity import install as install_pipeline_integrity
from rag_project.intelligence.production_contract_v2 import install as install_production_contract

RUNTIME_COMPOSITION_VERSION = "2026-09-13-composition-v1"
RUNTIME_PREPARED_ENV = "BOOKRAG_RUNTIME_PREPARED_VERSION"


def load_local_env(project_root: Path) -> None:
    """Load local .env values without overriding explicitly supplied process env."""
    env_file = project_root / ".env"
    if not env_file.is_file():
        return
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def normalize_runtime_environment() -> dict[str, Any]:
    """Clamp expensive local-runtime settings to the supported laptop envelope."""
    values = {
        "EMBEDDING_BATCH_SIZE": _bounded_int("EMBEDDING_BATCH_SIZE", 16, 16, 32),
        "EMBEDDING_RETRIES": _bounded_int("EMBEDDING_RETRIES", 2, 1, 3),
        "EMBEDDING_TIMEOUT_SECONDS": _bounded_float("EMBEDDING_TIMEOUT_SECONDS", 180.0, 30.0, 300.0),
    }
    for name, value in values.items():
        os.environ[name] = str(value)
    return values


def install_production_contracts() -> dict[str, Any]:
    """Install and report the authoritative runtime contracts in one order."""
    pipeline = install_pipeline_integrity()
    production = install_production_contract()
    ingestion = install_ingestion_contract()
    canonical = install_canonical_runtime()
    return {
        "composition_version": RUNTIME_COMPOSITION_VERSION,
        "pipeline_integrity": pipeline,
        "production_contract": production,
        "ingestion_contract": ingestion,
        "canonical_runtime": canonical,
    }


def prepare_runtime(project_root: Path) -> dict[str, Any]:
    """Prepare process configuration and all production contracts before app startup."""
    load_local_env(project_root)
    environment = normalize_runtime_environment()
    contracts = install_production_contracts()
    os.environ[RUNTIME_PREPARED_ENV] = RUNTIME_COMPOSITION_VERSION
    return {
        "composition_version": RUNTIME_COMPOSITION_VERSION,
        "environment": environment,
        "contracts": contracts,
        "prepared_marker": RUNTIME_PREPARED_ENV,
    }


def runtime_is_prepared() -> bool:
    """Return whether this process has been prepared by this exact composition contract."""
    return os.getenv(RUNTIME_PREPARED_ENV) == RUNTIME_COMPOSITION_VERSION


__all__ = [
    "RUNTIME_COMPOSITION_VERSION",
    "RUNTIME_PREPARED_ENV",
    "install_production_contracts",
    "load_local_env",
    "normalize_runtime_environment",
    "prepare_runtime",
    "runtime_is_prepared",
]
