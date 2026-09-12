"""Contract tests for the authoritative 17-phase diagnostic system.

These tests are deliberately deterministic and never launch the full suite. They verify
that the diagnostic control plane itself cannot silently lose phases, dependencies,
root-cause evidence, or bounded probe behavior.
"""
from __future__ import annotations

from rag_project.testing.deep_diagnostics import (
    PhaseResult,
    _metamorphic,
    _resource_probe,
    _mutation_probe,
    build_architecture,
    compress_cascade,
    fingerprint_failures,
)
from rag_project.testing.runner import PHASES


def test_exactly_seventeen_authoritative_phases_exist() -> None:
    assert [phase.number for phase in PHASES] == list(range(1, 18))
    assert len(PHASES) == 17
    assert len({phase.key for phase in PHASES}) == 17


def test_phase_dependency_graph_is_forward_only_and_in_range() -> None:
    for phase in PHASES:
        assert all(1 <= dep < phase.number for dep in phase.dependencies)


def test_phase_eight_is_a_real_metamorphic_probe() -> None:
    phase = PHASES[7]
    result = _metamorphic(phase)
    assert result.status == "PASS"
    assert result.details["stable_under_whitespace_and_case"] is True


def test_phase_eleven_is_non_destructive_mutation_analysis() -> None:
    phase = PHASES[10]
    result = _mutation_probe(phase)
    assert result.status in {"PASS", "WARN"}
    assert result.details["strategy"].startswith("shadow mutations only")
    assert "drop_metadata" in result.details["mutation_classes"]


def test_phase_fifteen_uses_a_bounded_resource_probe() -> None:
    phase = PHASES[14]
    result = _resource_probe(phase)
    assert result.status == "PASS"
    assert result.details["probe_iterations"] == 4
    assert result.details["peak_bytes"] >= result.details["current_bytes"]


def test_architecture_map_covers_major_rag_domains() -> None:
    architecture = build_architecture()
    expected = {"ingestion", "chunking", "embeddings", "retrieval", "intelligence", "generation", "storage", "evaluation"}
    assert expected.issubset(architecture["domains"])
    assert architecture["python_modules"] > 0
    assert architecture["test_files"] > 0


def test_failure_fingerprints_collapse_repeated_downstream_symptoms() -> None:
    phases = [
        PhaseResult(
            5,
            "cross_layer_invariants",
            "Cross-layer invariants",
            status="FAIL",
            failures=[
                {
                    "location": "rag_project/storage/vector_store.py:42",
                    "exception": "AssertionError",
                    "message": "document_id metadata was dropped",
                }
            ],
        ),
        PhaseResult(
            9,
            "retrieval_microscope",
            "Retrieval",
            status="FAIL",
            failures=[
                {
                    "location": "rag_project/retrieval/hybrid_retriever.py:88",
                    "exception": "AssertionError",
                    "message": "document_id metadata missing from ranked result",
                }
            ],
        ),
        PhaseResult(
            10,
            "rag_causality",
            "RAG causality",
            status="FAIL",
            failures=[
                {
                    "location": "rag_project/generation/final_answer_contract.py:31",
                    "exception": "AssertionError",
                    "message": "citation source metadata missing",
                }
            ],
        ),
    ]
    causes = fingerprint_failures(phases)
    assert causes
    assert causes[0].phase == 5
    assert set(causes[0].affected_phases) == {5, 9, 10}

    cascade = compress_cascade(phases, causes)
    assert cascade["failed_phases"] == [5, 9, 10]
    assert cascade["unique_root_causes"] == 1
    assert cascade["fix_order"][0]["phase"] == 5
