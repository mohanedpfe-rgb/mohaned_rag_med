"""Executable contract tests for the authoritative 17-phase diagnostic system."""
from __future__ import annotations

from rag_project.testing.deep_diagnostics import PhaseResult, build_architecture
from rag_project.testing.runner import PHASES
from rag_project.testing.advanced_phases import (
    adversarial_documents,
    certification_phase,
    contract_triangulation,
    cross_layer_invariants,
    diagnostic_chain,
    fingerprint_failures,
    cascade_compression,
    golden_benchmark,
    information_loss,
    metamorphic,
    mutation_detection,
    performance,
    rag_causality,
    resources,
    retrieval_microscope,
)


def _phase(number: int):
    return PHASES[number - 1]


def test_exactly_seventeen_authoritative_phases_exist() -> None:
    assert [phase.number for phase in PHASES] == list(range(1, 18))
    assert len({phase.key for phase in PHASES}) == 17


def test_dependency_graph_is_forward_only() -> None:
    for phase in PHASES:
        assert all(1 <= dependency < phase.number for dependency in phase.dependencies)


def test_all_concrete_phase_probes_are_callable() -> None:
    probes = {
        3: diagnostic_chain,
        4: contract_triangulation,
        5: cross_layer_invariants,
        6: information_loss,
        7: adversarial_documents,
        8: metamorphic,
        9: retrieval_microscope,
        10: rag_causality,
        11: mutation_detection,
        14: performance,
        15: resources,
        16: golden_benchmark,
    }
    for number, probe in probes.items():
        result = probe(_phase(number))
        assert result.status != "NOT_RUN", (number, result)
        assert result.details, number


def test_phase_eleven_reports_real_executed_mutants() -> None:
    result = mutation_detection(_phase(11))
    assert result.details["strategy"].startswith("executable shadow mutants")
    assert result.details["mutants_applicable"] >= 1
    assert result.details["mutants_killed"] == result.details["mutants_applicable"]
    assert result.details["kill_score"] == 1.0


def test_phase_fifteen_exercises_real_rag_stages() -> None:
    result = resources(_phase(15))
    assert result.status == "PASS"
    assert result.details["repetitions"] == 8
    assert set(result.details["pipeline_exercised"]) >= {
        "SemanticChunker.chunk_pages",
        "VectorStore.add_documents",
        "VectorStore.search_lexical",
        "VectorStore.search",
    }
    assert result.details["peak_bytes"] >= result.details["current_bytes"]


def test_phase_twelve_thirteen_and_seventeen_are_real_analysis_stages() -> None:
    failures = [
        PhaseResult(5, "cross_layer_invariants", "Cross-layer", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py:1", "exception": "IdentityConservationFailure", "message": "document_id dropped"}]),
        PhaseResult(9, "retrieval_microscope", "Retrieval", status="FAIL", failures=[{"location": "rag_project/retrieval/hybrid_retriever.py:2", "exception": "IdentityConservationFailure", "message": "document_id dropped"}]),
        PhaseResult(10, "rag_causality", "Causality", status="FAIL", failures=[{"location": "rag_project/retrieval/context_builder.py:3", "exception": "IdentityConservationFailure", "message": "document_id dropped"}]),
    ]
    fingerprints = fingerprint_failures(failures)
    cascade = cascade_compression(failures, fingerprints)
    assert fingerprints
    assert cascade["first_failed_phase"] == 5
    assert cascade["failed_phases"] == [5, 9, 10]

    results = {phase.number: PhaseResult(phase.number, phase.key, phase.name, status="PASS") for phase in PHASES}
    certified = certification_phase(_phase(17), results)
    assert certified.status == "PASS"
    assert certified.details["implementation_coverage"] == "17/17"


def test_architecture_map_still_covers_major_rag_domains() -> None:
    architecture = build_architecture()
    expected = {"ingestion", "chunking", "embeddings", "retrieval", "intelligence", "generation", "storage", "evaluation"}
    assert expected.issubset(architecture["domains"])
    assert architecture["python_modules"] > 0
    assert architecture["test_files"] > 0
