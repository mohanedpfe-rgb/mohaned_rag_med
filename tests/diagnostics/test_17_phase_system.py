"""Executable contract tests for the authoritative 17-phase diagnostic system."""
from __future__ import annotations

from rag_project.testing.deep_diagnostics import PhaseResult, build_architecture
from rag_project.testing.runner import PHASES
from rag_project.testing.advanced_phases import (
    contract_triangulation,
    cross_layer_invariants,
    diagnostic_chain,
    fingerprint_failures,
    cascade_compression,
    metamorphic,
)
from rag_project.testing.robust_probes import information_loss, retrieval_microscope
from rag_project.testing.strict_phases import (
    phase10_production_generation,
    phase11_mutation_testing,
    phase13_causal_graph,
    phase17_strict_certification,
    wrap_phase7,
    wrap_phase14,
    wrap_phase15,
)
from rag_project.testing.final_probes import phase16_independent_gold


def _phase(number: int):
    return PHASES[number - 1]


def test_exactly_seventeen_authoritative_phases_exist() -> None:
    assert [phase.number for phase in PHASES] == list(range(1, 18))
    assert len({phase.key for phase in PHASES}) == 17


def test_dependency_graph_is_forward_only() -> None:
    for phase in PHASES:
        assert all(1 <= dependency < phase.number for dependency in phase.dependencies)


def test_strict_phase_probes_are_executable() -> None:
    probes = {
        3: diagnostic_chain,
        4: contract_triangulation,
        5: cross_layer_invariants,
        6: information_loss,
        7: wrap_phase7,
        8: metamorphic,
        9: retrieval_microscope,
        10: phase10_production_generation,
        11: phase11_mutation_testing,
        14: wrap_phase14,
        15: wrap_phase15,
        16: phase16_independent_gold,
    }
    for number, probe in probes.items():
        result = probe(_phase(number))
        assert result.status != "NOT_RUN", (number, result)
        assert result.details, number


def test_phase_ten_exercises_production_generation_and_verification() -> None:
    result = phase10_production_generation(_phase(10))
    assert result.status == "PASS", result.failures
    assert result.details["answer_generated"] is True
    assert result.details["citations_present"] is True
    assert result.details["citation_ids_valid"] is True
    assert result.details["verification_allow"] is True


def test_phase_eleven_kills_mutants_with_pytest() -> None:
    result = phase11_mutation_testing(_phase(11))
    assert result.details["strategy"].startswith("executable source mutants + real pytest")
    assert result.details["mutants_applicable"] >= 1
    assert result.details["mutants_killed"] == result.details["mutants_applicable"]
    assert result.details["kill_score"] == 1.0


def test_phase_six_measures_representation_survival() -> None:
    result = information_loss(_phase(6))
    assert "token_survival" in result.details
    assert "field_survival" in result.details
    assert result.details["representation_chain"] == ["PageExtraction", "Chunk", "VectorStore", "SQLite lexical"]


def test_phase_nine_measures_retrieval_metrics_and_filtering() -> None:
    result = retrieval_microscope(_phase(9))
    assert "lexical_recall_at_3" in result.details
    assert "semantic_recall_at_3" in result.details
    assert result.details["metadata_filter_correct"] is True


def test_phase_thirteen_builds_evidence_backed_failure_graph() -> None:
    failures = {
        5: PhaseResult(5, "cross_layer_invariants", "Cross-layer", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document_id dropped"}]),
        9: PhaseResult(9, "retrieval_microscope", "Retrieval", status="FAIL", failures=[{"location": "rag_project/storage/vector_store.py", "exception": "IdentityConservationFailure", "message": "document_id dropped"}]),
    }
    result = phase13_causal_graph(_phase(13), failures)
    assert result.status == "PASS"
    assert result.details["nodes"]
    assert result.details["edges"]
    assert result.details["candidate_roots"]


def test_phase_sixteen_uses_separate_independent_corpus_and_labels() -> None:
    result = phase16_independent_gold(_phase(16))
    assert result.status == "PASS", result.failures
    assert result.details["gold_labels_independent_of_corpus_text"] is True
    assert result.details["corpus_document_count"] >= 5
    assert result.details["retrieval_recall"] >= 0.8


def test_phase_seventeen_rejects_fake_pass_matrix() -> None:
    fake = {phase.number: PhaseResult(phase.number, phase.key, phase.name, status="PASS", details={}) for phase in PHASES}
    certified = phase17_strict_certification(_phase(17), fake)
    assert certified.status == "FAIL"
    assert certified.details["implementation_coverage"] != "17/17"
    assert certified.details["evidence_failures"]


def test_phase_twelve_and_cascade_still_report_failures() -> None:
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


def test_architecture_map_still_covers_major_rag_domains() -> None:
    architecture = build_architecture()
    expected = {"ingestion", "chunking", "embeddings", "retrieval", "intelligence", "generation", "storage", "evaluation"}
    assert expected.issubset(architecture["domains"])
    assert architecture["python_modules"] > 0
    assert architecture["test_files"] > 0
