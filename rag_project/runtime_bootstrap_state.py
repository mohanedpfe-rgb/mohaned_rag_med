"""Neutral process state for the production composition lifecycle.

This module is deliberately below both the composition root and the
application service. It carries no UI or installer dependencies, preventing
the application layer from importing the composition root merely to inspect
bootstrap state.
"""
from __future__ import annotations

import os

RUNTIME_COMPOSITION_VERSION = "2026-09-13-composition-v1"
RUNTIME_PREPARED_ENV = "BOOKRAG_RUNTIME_PREPARED_VERSION"


def mark_prepared() -> None:
    """Mark this process as prepared for the exact supported composition version."""
    os.environ[RUNTIME_PREPARED_ENV] = RUNTIME_COMPOSITION_VERSION


def is_prepared() -> bool:
    """Return true only when the process marker matches this exact contract version."""
    return os.getenv(RUNTIME_PREPARED_ENV) == RUNTIME_COMPOSITION_VERSION


__all__ = ["RUNTIME_COMPOSITION_VERSION", "RUNTIME_PREPARED_ENV", "is_prepared", "mark_prepared"]
