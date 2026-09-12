from __future__ import annotations

from rag_project.testing.implementation_contracts import validate_runtime_ownership
from rag_project.testing.runner import PHASES


def test_strong_phase_wiring_is_authoritative() -> None:
    report = validate_runtime_ownership()
    assert report["pass"], report["failures"]
    rows = {row["phase"]: row for row in report["ownership_rows"]}
    expected = {
        3: "strict_diagnostic_chain",
        4: "strict_contract_triangulation",
        5: "strict_cross_layer_invariants",
        6: "strict_information_loss",
        8: "run_full_metamorphic_suite",
        11: "run_full_mutation_suite",
        13: "strict_causal_phase",
        15: "strict_resource_stability",
    }
    for phase, qualname in expected.items():
        assert rows[phase]["actual_qualname"] == qualname
        assert rows[phase]["status"] == "PASS"


def test_phase_registry_remains_exactly_seventeen() -> None:
    assert [phase.number for phase in PHASES] == list(range(1, 18))
