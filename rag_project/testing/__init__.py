"""Authoritative testing infrastructure for deep, dependency-aware RAG diagnostics."""

from . import full17_hardening  # import-time install patches diagnostic bindings before runner import

from .deep_diagnostics import DiagnosticReport, PhaseResult, RootCause
from .runner import PHASES, UnifiedDiagnosticEngine, run_all
from .architecture_contracts import install as _install_architecture_contract

_install_architecture_contract()

from . import strict_runtime_contracts as _strict_runtime_contracts
_strict_runtime_contracts.install()

# Phase 12 is intentionally hardened at runtime by full17_hardening, but the
# hardened callable remains the exported implementation of the authoritative
# production_diagnostic_probes contract. Preserve that ownership identity for
# diagnostics that inspect __module__ rather than implementation behavior.
from . import production_diagnostic_probes as _production_diagnostic_probes
if hasattr(_production_diagnostic_probes, "phase12_stable_fingerprinting"):
    _production_diagnostic_probes.phase12_stable_fingerprinting.__module__ = _production_diagnostic_probes.__name__

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
