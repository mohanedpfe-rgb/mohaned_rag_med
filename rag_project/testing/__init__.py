"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from . import full17_hardening  # import-time install patches diagnostic bindings before runner import

from .deep_diagnostics import DiagnosticReport, PhaseResult, RootCause
from .runner import PHASES, UnifiedDiagnosticEngine, run_all
from .architecture_contracts import install as _install_architecture_contract

_install_architecture_contract()

from . import strict_runtime_contracts as _strict_runtime_contracts
_strict_runtime_contracts.install()

# Final certification is deliberately a pure boundary: it validates the
# already-executed 1..16 results and never re-runs phases inside Phase 17.
from .strict_phase17_final import phase17_strict_completion as _strict_phase17_final
from .runner import UnifiedDiagnosticEngine as _UnifiedDiagnosticEngine
_UnifiedDiagnosticEngine._execute.__globals__["phase17_strict_completion"] = _strict_phase17_final

__all__ = [
    "PHASES",
    "DiagnosticReport",
    "PhaseResult",
    "RootCause",
    "UnifiedDiagnosticEngine",
    "run_all",
]
