from __future__ import annotations

from rag_project.testing.deep_diagnostics import PhaseResult
from rag_project.testing.runner import PHASES
from rag_project.testing.strict_phase17_final import phase17_strict_completion


def _placeholder_results() -> dict[int, PhaseResult]:
    results: dict[int, PhaseResult] = {}
    for phase in PHASES[:16]:
        results[phase.number] = PhaseResult(
            phase.number,
            phase.key,
            phase.name,
            status="FAIL",
            details={"evidence_level": "boundary-test"},
        )
    return results


def test_phase17_is_pure_certification_boundary(monkeypatch) -> None:
    import rag_project.testing.full_metamorphic_probes as metamorphic
    import rag_project.testing.full_mutation_probes as mutation

    def forbidden(*args, **kwargs):
        raise AssertionError("Phase 17 must not re-execute an earlier diagnostic phase")

    monkeypatch.setattr(metamorphic, "run_full_metamorphic_suite", forbidden)
    monkeypatch.setattr(mutation, "run_full_mutation_suite", forbidden)

    result = phase17_strict_completion(PHASES[16], _placeholder_results())

    assert result.status == "FAIL"
    assert result.details["reexecuted_phases"] == []
    assert result.details["authoritative_recheck_phase_identity"]["verified"] is True
