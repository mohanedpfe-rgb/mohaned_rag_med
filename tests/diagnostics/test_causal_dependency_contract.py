from __future__ import annotations

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.strict_causal_phase import _strict_dependency_graph, _phase_is_ancestor
from rag_project.testing.runner import PHASES


def test_transitive_phase_dependency_is_recognized() -> None:
    assert _phase_is_ancestor(tuple(PHASES), 5, 10) is True
    assert _phase_is_ancestor(tuple(PHASES), 5, 9) is True


def test_same_failure_without_dependency_cannot_create_causal_edge() -> None:
    pair = next(
        (left, right)
        for left in PHASES
        for right in PHASES
        if right.number > left.number and left.number not in set(right.dependencies) and not _phase_is_ancestor(tuple(PHASES), left.number, right.number)
    )
    left, right = pair
    failures = {
        left.number: PhaseResult(left.number, left.key, left.name, status="FAIL", failures=[{
            "location": "rag_project/shared_module.py:10",
            "exception": "SharedFailure",
            "message": "same controlled failure",
        }]),
        right.number: PhaseResult(right.number, right.key, right.name, status="FAIL", failures=[{
            "location": "rag_project/shared_module.py:20",
            "exception": "SharedFailure",
            "message": "same controlled failure",
        }]),
    }
    edges, _ = _strict_dependency_graph(PHASES[12], failures)
    assert not edges
