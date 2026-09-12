from rag_project.testing.implementation_contracts import validate_runtime_ownership
from rag_project.testing.runner import PHASES


def test_foundation_phases_use_fault_sensitive_authoritative_bindings():
    report = validate_runtime_ownership()
    assert report["pass"], report["failures"]
    rows = {row["phase"]: row for row in report["ownership_rows"]}
    assert rows[3]["actual_qualname"] == "strict_diagnostic_chain"
    assert rows[4]["actual_qualname"] == "strict_contract_triangulation"
    assert rows[5]["actual_qualname"] == "strict_cross_layer_invariants"
    assert rows[6]["actual_qualname"] == "strict_information_loss"


def test_phase_registry_remains_exactly_seventeen():
    assert [phase.number for phase in PHASES] == list(range(1, 18))
