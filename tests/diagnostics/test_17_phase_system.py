"""Executable contract tests for the authoritative 17-phase diagnostic system."""
from __future__ import annotations

from rag_project.testing.deep_diagnostics import PhaseResult, build_architecture
from rag_project.testing.implementation_contracts import validate_runtime_ownership
from rag_project.testing.runner import PHASES, _base_phase17_strict, phase17_strict_completion


def _phase(number: int):
    return PHASES[number - 1]


def test_exactly_seventeen_authoritative_phases_exist() -> None:
    assert [phase.number for phase in PHASES] == list(range(1, 18))
    assert len({phase.key for phase in PHASES}) == 17


def test_dependency_graph_is_forward_only() -> None:
    for phase in PHASES:
        assert all(1 <= dependency < phase.number for dependency in phase.dependencies)


def test_authoritative_runtime_ownership_is_complete() -> None:
    report = validate_runtime_ownership()
    assert report["phase_count"] == 17
    assert report["phase_numbers"] == list(range(1, 18))
    assert report["dispatch_numbers"] == list(range(1, 18))
    assert report["hardened_runtime_bindings_verified"] is True
    assert report["pass"] is True, report["failures"]


def test_phase17_rejects_fake_pass_matrix() -> None:
    fake = {
        phase.number: PhaseResult(
            phase.number,
            phase.key,
            phase.name,
            status="PASS",
            details={"fake": True},
        )
        for phase in PHASES
        if phase.number < 17
    }
    certified = _base_phase17_strict(_phase(17), fake)
    assert certified.status == "FAIL"
    assert certified.details.get("implementation_coverage") != "17/17"
    assert certified.details.get("evidence_failures")


def test_phase17_rechecks_preserve_phase_identity(monkeypatch) -> None:
    from rag_project.testing import full_metamorphic_probes, full_mutation_probes

    def fake_suite(phase):
        return PhaseResult(phase.number, phase.key, phase.name, status="PASS", details={"evidence_level": "synthetic_test"})

    monkeypatch.setattr(full_metamorphic_probes, "run_full_metamorphic_suite", fake_suite)
    monkeypatch.setattr(full_mutation_probes, "run_full_mutation_suite", fake_suite)
    results = {
        phase.number: PhaseResult(phase.number, phase.key, phase.name, status="FAIL", details={"evidence_level": "synthetic"})
        for phase in PHASES
        if phase.number < 17
    }
    phase17_strict_completion(_phase(17), results)
    assert results[8].number == 8
    assert results[11].number == 11


def test_architecture_map_covers_all_major_rag_domains() -> None:
    architecture = build_architecture()
    expected = {
        "ingestion",
        "chunking",
        "embeddings",
        "retrieval",
        "intelligence",
        "generation",
        "storage",
        "evaluation",
    }
    assert expected.issubset(architecture["domains"])
    assert architecture["python_modules"] > 0
    assert architecture["test_files"] > 0
