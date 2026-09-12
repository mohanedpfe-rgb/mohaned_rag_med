"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from . import full17_hardening  # import-time install patches diagnostic bindings before runner import

from .deep_diagnostics import DiagnosticReport, PhaseResult, RootCause
from .runner import PHASES, UnifiedDiagnosticEngine, run_all
from .architecture_contracts import install as _install_architecture_contract

_install_architecture_contract()

from . import strict_runtime_contracts as _strict_runtime_contracts
_strict_runtime_contracts.install()

# Phase 17 is imported directly by runner.py as the sole certification authority.
# No package-level rebinding or legacy wrapper is permitted here.

__all__ = [
    "PHASES",
    "DiagnosticReport",
    "PhaseResult",
    "RootCause",
    "UnifiedDiagnosticEngine",
    "run_all",
]
