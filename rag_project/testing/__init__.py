"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from .full17_hardening import install as _install_full17_hardening
_install_full17_hardening()

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
