from __future__ import annotations

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.production_diagnostic_probes import (
    phase12_stable_fingerprinting,
    phase13_known_causal_graph,
)
from rag_project.testing.runner import PHASES


def test_phase_twelve_fingerprint_self_tests_are_executable() -> None:
    result = phase12_stable_fingerprinting(PHASES[11], {})
    assert result.status == "PASS", result.failures
    assert result.details["self_test_equivalent_inputs_same"] is True
    assert result.details["self_test_different_module_different"] is True


def test_phase_thirteen_causal_self_test_is_executable() -> None:
    results = {
        5: PhaseResult(5, "cross_layer_invariants", "Cross-layer", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before retrieval"}]),
        9: PhaseResult(9, "retrieval_microscope", "Retrieval", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document identity dropped before ranking"}]),
    }
    result = phase13_known_causal_graph(PHASES[12], results)
    assert result.status == "PASS", result.failures
    assert result.details["known_causal_fixture_verified"] is True
