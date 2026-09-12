"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from . import full17_hardening  # import-time install patches diagnostic bindings before runner import

from .deep_diagnostics import DiagnosticReport, PhaseResult, RootCause
from .runner import PHASES, UnifiedDiagnosticEngine, run_all

__all__ = [
    "PHASES",
    "DiagnosticReport",
    "PhaseResult",
    "RootCause",
    "UnifiedDiagnosticEngine",
    "run_all",
]
