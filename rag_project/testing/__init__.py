"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from . import full17_hardening  # import-time install patches diagnostic bindings before runner import

from .deep_diagnostics import DiagnosticReport, PhaseResult, RootCause
from .runner import PHASES, UnifiedDiagnosticEngine, run_all
from .architecture_contracts import install as _install_architecture_contract

# Install Phase 1 ownership/dependency enforcement after the runner and its
# production diagnostic module have been loaded, so the authoritative Phase 17
# semantic contract uses the hardened implementation.
_install_architecture_contract()

__all__ = [
    "PHASES",
    "DiagnosticReport",
    "PhaseResult",
    "RootCause",
    "UnifiedDiagnosticEngine",
    "run_all",
]
