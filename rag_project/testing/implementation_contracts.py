"""Explicit implementation ownership and runtime wiring contract for the 17-phase doctor."""
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
    PhaseOwnership(2, "fast_health", "strict_fast_health", "rag_project.testing.architecture_contracts", "runtime_binding"),
    PhaseOwnership(3, "diagnostic_chain", "strict_diagnostic_chain", "rag_project.testing.strict_foundation_phases", "runtime_binding"),
    PhaseOwnership(4, "contract_triangulation", "strict_contract_triangulation", "rag_project.testing.strict_foundation_phases", "runtime_binding"),
    PhaseOwnership(5, "cross_layer_invariants", "strict_cross_layer_invariants", "rag_project.testing.strict_foundation_phases", "runtime_binding"),
    PhaseOwnership(6, "information_loss", "strict_information_loss", "rag_project.testing.strict_information_loss", "runtime_binding"),
    PhaseOwnership(7, "adversarial_documents", "phase7_production_pdf_lab", "rag_project.testing.production_document_probes", "runtime_dispatch"),
    PhaseOwnership(8, "metamorphic", "run_full_metamorphic_suite", "rag_project.testing.full_metamorphic_probes", "runtime_binding"),
    PhaseOwnership(9, "retrieval_microscope", "phase9_independent_retrieval", "rag_project.testing.production_retrieval_probes", "runtime_dispatch"),
    PhaseOwnership(10, "rag_causality", "phase10_canonical_answer_engine", "rag_project.testing.production_answer_probes", "runtime_dispatch"),
    PhaseOwnership(11, "mutation", "run_full_mutation_suite", "rag_project.testing.full_mutation_probes", "runtime_binding"),
    PhaseOwnership(12, "fingerprinting", "phase12_stable_fingerprinting", "rag_project.testing.production_diagnostic_probes", "runtime_dispatch"),
    PhaseOwnership(13, "cascade", "strict_causal_phase", "rag_project.testing.strict_causal_phase", "runtime_binding"),
    PhaseOwnership(14, "performance", "phase14_production_benchmark", "rag_project.testing.production_benchmark_probes", "runtime_dispatch"),
    PhaseOwnership(15, "resources", "strict_resource_stability", "rag_project.testing.strict_runtime_contracts", "runtime_binding"),
    PhaseOwnership(16, "golden_benchmark", "strict_phase16_production_ingestion_benchmark", "rag_project.testing.strict_phase16_production", "runtime_binding"),
    PhaseOwnership(17, "certification", "phase17_strict_completion", "rag_project.testing.strict_phase17_final", "runtime_binding"),
)


def _qualname(value: Any) -> str:
    return str(getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)))


def _module(value: Any) -> str:
    return str(getattr(value, "__module__", "unknown"))


def _source_dispatch_numbers(runner: Any) -> set[int]:
    tree = ast.parse(inspect.getsource(runner.UnifiedDiagnosticEngine._execute))
    numbers: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) and node.left.attr == "number" and len(node.comparators) == 1:
            literal = node.comparators[0]
            if isinstance(literal, ast.Constant) and isinstance(literal.value, int):
                numbers.add(literal.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, int):
                    numbers.add(key.value)
    return numbers


def validate_runtime_ownership() -> dict[str, Any]:
    from rag_project.testing import runner, deep_diagnostics
    from rag_project.testing.architecture_contracts import strict_fast_health
    from rag_project.testing.full_metamorphic_probes import run_full_metamorphic_suite
    from rag_project.testing.full_mutation_probes import run_full_mutation_suite
    from rag_project.testing.production_answer_probes import phase10_canonical_answer_engine
    from rag_project.testing.production_benchmark_probes import phase14_production_benchmark
    from rag_project.testing.production_diagnostic_probes import phase12_stable_fingerprinting
    from rag_project.testing.production_document_probes import phase7_production_pdf_lab
    from rag_project.testing.production_retrieval_probes import phase9_independent_retrieval
    from rag_project.testing.strict_phase16_production import strict_phase16_production_ingestion_benchmark
    from rag_project.testing.strict_phase17_final import phase17_strict_completion
    from rag_project.testing.strict_runtime_contracts import strict_resource_stability
    from rag_project.testing.strict_information_loss import strict_information_loss
    from rag_project.testing.strict_foundation_phases import strict_diagnostic_chain, strict_contract_triangulation, strict_cross_layer_invariants
    from rag_project.testing.strict_causal_phase import strict_causal_phase

    globals_map = runner.UnifiedDiagnosticEngine._execute.__globals__
    resolved = {
        1: (runner.UnifiedDiagnosticEngine._execute, "self-dispatch"),
        2: (globals_map.get("strict_fast_health"), "global binding"),
        3: (globals_map.get("strict_diagnostic_chain"), "global binding"),
        4: (globals_map.get("strict_contract_triangulation"), "global binding"),
        5: (globals_map.get("strict_cross_layer_invariants"), "global binding"),
        6: (globals_map.get("strict_information_loss"), "global binding"),
        7: (globals_map.get("phase7_production_pdf_lab"), "global binding"),
        8: (globals_map.get("run_full_metamorphic_suite"), "global binding"),
        9: (globals_map.get("phase9_independent_retrieval"), "global binding"),
        10: (globals_map.get("phase10_canonical_answer_engine"), "global binding"),
        11: (globals_map.get("run_full_mutation_suite"), "global binding"),
        12: (globals_map.get("phase12_stable_fingerprinting"), "global binding"),
        13: (globals_map.get("strict_causal_phase"), "global binding"),
        14: (globals_map.get("phase14_production_benchmark"), "global binding"),
        15: (globals_map.get("strict_resource_stability"), "global binding"),
        16: (globals_map.get("strict_phase16_production_ingestion_benchmark"), "global binding"),
        17: (globals_map.get("phase17_strict_completion"), "global binding"),
    }
    expected = {
        2: strict_fast_health,
        3: strict_diagnostic_chain,
        4: strict_contract_triangulation,
        5: strict_cross_layer_invariants,
        6: strict_information_loss,
        7: phase7_production_pdf_lab,
        8: run_full_metamorphic_suite,
        9: phase9_independent_retrieval,
        10: phase10_canonical_answer_engine,
        11: run_full_mutation_suite,
        12: phase12_stable_fingerprinting,
        13: strict_causal_phase,
        14: phase14_production_benchmark,
        15: strict_resource_stability,
        16: strict_phase16_production_ingestion_benchmark,
        17: phase17_strict_completion,
    }
    failures: list[dict[str, Any]] = []
    phase_numbers = [p.number for p in runner.PHASES]
    if phase_numbers != list(range(1, 18)) or len(set(phase_numbers)) != 17:
        failures.append({"reason": "phase registry is not exactly 1..17", "actual": phase_numbers})
    try:
        dispatch_numbers = _source_dispatch_numbers(runner)
    except Exception as exc:
        dispatch_numbers = set()
        failures.append({"reason": "unable to statically inspect runner dispatch", "exception": type(exc).__name__})
    missing = set(range(1, 18)) - dispatch_numbers
    if missing:
        failures.append({"reason": "runner dispatch source missing phase numbers", "missing": sorted(missing)})

    rows = []
    for owner in EXPECTED_OWNERS:
        value, binding_kind = resolved[owner.number]
        actual_qualname = _qualname(value) if value is not None else None
        actual_module = _module(value) if value is not None else None
        row = {"phase": owner.number, "key": owner.key, "expected_symbol": owner.authoritative_symbol, "actual_qualname": actual_qualname, "actual_module": actual_module, "binding_kind": binding_kind, "expected_module": owner.module, "expected_evidence_kind": owner.evidence_kind, "status": "PASS"}
        if value is None:
            row["status"] = "FAIL"
            failures.append({"phase": owner.number, "reason": "authoritative callable is missing", "expected": owner.authoritative_symbol})
        elif owner.number in expected and value is not expected[owner.number]:
            row["status"] = "FAIL"
            failures.append({"phase": owner.number, "reason": "incorrect runtime authoritative callable", "expected_qualname": _qualname(expected[owner.number]), "actual_qualname": actual_qualname})
        elif actual_module != owner.module:
            row["status"] = "FAIL"
            failures.append({"phase": owner.number, "reason": "authoritative callable resolved from unexpected module", "expected_module": owner.module, "actual_module": actual_module})
        rows.append(row)

    return {"contract_version": "17-phase-implementation-ownership-v7", "phase_count": 17, "phase_numbers": phase_numbers, "dispatch_numbers": sorted(dispatch_numbers), "expected_dispatch_numbers": list(range(1, 18)), "hardened_runtime_bindings_verified": all(resolved[n][0] is expected[n] for n in expected), "ownership_rows": rows, "failures": failures, "pass": not failures}


__all__ = ["EXPECTED_OWNERS", "PhaseOwnership", "validate_runtime_ownership"]