"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from . import full17_hardening  # import-time install patches diagnostic bindings before runner import

from .deep_diagnostics import DiagnosticReport, PhaseResult, RootCause
from .runner import PHASES, UnifiedDiagnosticEngine, run_all
from .architecture_contracts import install as _install_architecture_contract

# Install Phase 1 ownership/dependency enforcement after the runner and its
# production diagnostic module have been loaded, so the authoritative Phase 17
# semantic contract uses the hardened implementation.
_install_architecture_contract()

# Install the final trend-aware resource contract after the runner exists so
# phase 15 cannot fall back to the legacy first-vs-last resource probe.
from . import strict_runtime_contracts as _strict_runtime_contracts
_strict_runtime_contracts.install()

__all__ = [
    "PHASES",
    "DiagnosticReport",
    "PhaseResult",
    "RootCause",
    "UnifiedDiagnosticEngine",
    "run_all",
]
