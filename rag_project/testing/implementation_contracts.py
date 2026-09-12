"""Explicit implementation ownership and runtime wiring contract for the 17-phase doctor.

This module deliberately does not execute any diagnostic phase itself. It verifies that the
runner exposes exactly one authoritative dispatch for each phase and that hardened runtime
bindings are the ones actually reachable when the package is imported.
"""
from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhaseOwnership:
    number: int
    key: str
    authoritative_symbol: str
    module: str
    evidence_kind: str


EXPECTED_OWNERS = (
    PhaseOwnership(1, "architecture_map", "UnifiedDiagnosticEngine._phase1", "rag_project.testing.runner", "runtime_dispatch"),
    PhaseOwnership(2, "fast_health", "deep_diagnostics._fast_health", "rag_project.testing.deep_diagnostics", "runtime_dispatch"),
    PhaseOwnership(3, "diagnostic_chain", "diagnostic_chain", "rag_project.testing.advanced_phases", "runtime_dispatch"),
    PhaseOwnership(4, "contract_triangulation", "contract_triangulation", "rag_project.testing.advanced_phases", "runtime_dispatch"),
    PhaseOwnership(5, "cross_layer_invariants", "cross_layer_invariants", "rag_project.testing.advanced_phases", "runtime_dispatch"),
    PhaseOwnership(6, "information_loss", "information_loss", "rag_project.testing.advanced_phases", "runtime_dispatch"),
    PhaseOwnership(7, "adversarial_documents", "phase7_production_pdf_lab", "rag_project.testing.production_document_probes", "runtime_dispatch"),
    PhaseOwnership(8, "metamorphic", "run_full_metamorphic_suite", "rag_project.testing.full_metamorphic_probes", "runtime_binding"),
    PhaseOwnership(9, "retrieval_microscope", "phase9_independent_retrieval", "rag_project.testing.production_retrieval_probes", "runtime_dispatch"),
    PhaseOwnership(10, "rag_causality", "phase10_canonical_answer_engine", "rag_project.testing.production_answer_probes", "runtime_dispatch"),
    PhaseOwnership(11, "mutation", "run_full_mutation_suite", "rag_project.testing.full_mutation_probes", "runtime_binding"),
    PhaseOwnership(12, "fingerprinting", "phase12_stable_fingerprinting", "rag_project.testing.production_diagnostic_probes", "runtime_dispatch"),
    PhaseOwnership(13, "cascade", "phase13_known_causal_graph", "rag_project.testing.production_diagnostic_probes", "runtime_dispatch"),
    PhaseOwnership(14, "performance", "phase14_production_benchmark", "rag_project.testing.production_benchmark_probes", "runtime_dispatch"),
    PhaseOwnership(15, "resources", "strict_resource_stability", "rag_project.testing.strict_runtime_contracts", "runtime_binding"),
    PhaseOwnership(16, "golden_benchmark", "phase16_production_ingestion_benchmark", "rag_project.testing.production_path_probes", "runtime_dispatch"),
    PhaseOwnership(17, "certification", "phase17_strict_completion", "rag_project.testing.production_diagnostic_probes", "runtime_binding"),
)


def _qualname(value: Any) -> str:
    return str(getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)))


def _module(value: Any) -> str:
    return str(getattr(value, "__module__", "unknown"))


def _source_dispatch_numbers(runner: Any) -> set[int]:
    source = inspect.getsource(runner.UnifiedDiagnosticEngine._execute)
    tree = ast.parse(source)
    numbers: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Attribute) and node.left.attr == "number"):
            continue
        if len(node.comparators) != 1:
            continue
        literal = node.comparators[0]
        if isinstance(literal, ast.Constant) and isinstance(literal.value, int):
            numbers.add(literal.value)
    return numbers


def validate_runtime_ownership() -> dict[str, Any]:
    from rag_project.testing import runner
    from rag_project.testing import deep_diagnostics
    from rag_project.testing.architecture_contracts import strict_fast_health
    from rag_project.testing.full_metamorphic_probes import run_full_metamorphic_suite
    from rag_project.testing.full_mutation_probes import run_full_mutation_suite
    from rag_project.testing.production_answer_probes import phase10_canonical_answer_engine
    from rag_project.testing.production_benchmark_probes import phase14_production_benchmark
    from rag_project.testing.production_diagnostic_probes import phase12_stable_fingerprinting, phase13_known_causal_graph
    from rag_project.testing.production_document_probes import phase7_production_pdf_lab
    from rag_project.testing.production_path_probes import phase16_production_ingestion_benchmark
    from rag_project.testing.production_retrieval_probes import phase9_independent_retrieval
    from rag_project.testing.strict_runtime_contracts import strict_resource_stability
    from rag_project.testing.advanced_phases import contract_triangulation, cross_layer_invariants, diagnostic_chain, information_loss

    globals_map = runner.UnifiedDiagnosticEngine._execute.__globals__
    resolved = {
        1: (runner.UnifiedDiagnosticEngine._execute, "self-dispatch"),
        2: (deep_diagnostics._fast_health, "deep_diagnostics binding"),
        3: (globals_map.get("diagnostic_chain"), "global binding"),
        4: (globals_map.get("contract_triangulation"), "global binding"),
        5: (globals_map.get("cross_layer_invariants"), "global binding"),
        6: (globals_map.get("information_loss"), "global binding"),
        7: (globals_map.get("phase7_production_pdf_lab"), "global binding"),
        8: (globals_map.get("metamorphic"), "global binding"),
        9: (globals_map.get("phase9_independent_retrieval"), "global binding"),
        10: (globals_map.get("phase10_canonical_answer_engine"), "global binding"),
        11: (globals_map.get("_hardened_mutation_phase"), "global binding"),
        12: (globals_map.get("phase12_stable_fingerprinting"), "global binding"),
        13: (globals_map.get("phase13_known_causal_graph"), "global binding"),
        14: (globals_map.get("phase14_production_benchmark"), "global binding"),
        15: (globals_map.get("phase15_resource_stability"), "global binding"),
        16: (globals_map.get("phase16_production_ingestion_benchmark"), "global binding"),
        17: (globals_map.get("phase17_strict_completion"), "global binding"),
    }
    expected_callables = {
        2: strict_fast_health,
        3: diagnostic_chain,
        4: contract_triangulation,
        5: cross_layer_invariants,
        6: information_loss,
        7: phase7_production_pdf_lab,
        8: run_full_metamorphic_suite,
        9: phase9_independent_retrieval,
        10: phase10_canonical_answer_engine,
        11: run_full_mutation_suite,
        12: phase12_stable_fingerprinting,
        13: phase13_known_causal_graph,
        14: phase14_production_benchmark,
        15: strict_resource_stability,
        16: phase16_production_ingestion_benchmark,
    }
    failures: list[dict[str, Any]] = []
    phase_numbers = [phase.number for phase in runner.PHASES]
    if phase_numbers != list(range(1, 18)):
        failures.append({"reason": "phase registry is not exactly 1..17", "actual": phase_numbers})
    if len(set(phase_numbers)) != 17:
        failures.append({"reason": "phase registry contains duplicates"})
    try:
        dispatch_numbers = _source_dispatch_numbers(runner)
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        dispatch_numbers = set()
        failures.append({"reason": "unable to statically inspect UnifiedDiagnosticEngine._execute", "exception": type(exc).__name__})
    expected_dispatch_numbers = set(range(1, 18))
    if not expected_dispatch_numbers.issubset(dispatch_numbers):
        failures.append({"reason": "runner dispatch source does not expose every phase number", "missing": sorted(expected_dispatch_numbers - dispatch_numbers)})

    rows = []
    for owner in EXPECTED_OWNERS:
        value, binding_kind = resolved[owner.number]
        row = {
            "phase": owner.number,
            "key": owner.key,
            "expected_symbol": owner.authoritative_symbol,
            "actual_qualname": _qualname(value) if value is not None else None,
            "actual_module": _module(value) if value is not None else None,
            "binding_kind": binding_kind,
            "expected_module": owner.module,
            "status": "PASS",
        }
        if value is None:
            row["status"] = "FAIL"
            failures.append({"phase": owner.number, "reason": "authoritative callable is missing", "expected": owner.authoritative_symbol})
        elif owner.number in expected_callables and value is not expected_callables[owner.number]:
            row["status"] = "FAIL"
            failures.append({"phase": owner.number, "reason": "runner is bound to a different callable than the declared authoritative implementation", "expected_qualname": _qualname(expected_callables[owner.number]), "actual_qualname": _qualname(value)})
        rows.append(row)

    p17 = resolved[17][0]
    if p17 is None or not getattr(p17, "_provenance_wrapped", False):
        failures.append({"phase": 17, "reason": "certification callable is not provenance wrapped"})
    else:
        rows[-1]["provenance_wrapped"] = True

    hardened_ok = all(resolved[number][0] is expected_callables[number] for number in expected_callables)
    return {
        "contract_version": "17-phase-implementation-ownership-v1",
        "phase_count": 17,
        "phase_numbers": phase_numbers,
        "dispatch_numbers": sorted(dispatch_numbers),
        "expected_dispatch_numbers": sorted(expected_dispatch_numbers),
        "hardened_runtime_bindings_verified": hardened_ok,
        "ownership_rows": rows,
        "failures": failures,
        "pass": not failures,
    }


__all__ = ["EXPECTED_OWNERS", "PhaseOwnership", "validate_runtime_ownership"]
