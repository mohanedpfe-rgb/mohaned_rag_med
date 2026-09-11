from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from rag_project.intelligence.advanced_rag_engine import (
    detect_contradictions,
    evidence_coverage,
    retrieve_document_aware,
    route_query,
)
from rag_project.intelligence.god_mode_100 import enhanced_god_answer


@dataclass
class Hit:
    doc_id: str
    text: str
    metadata: dict
    score: float
    vector_score: float = 0.8
    lexical_score: float = 0.8


def test_query_router_specializes_summary_table_and_comparison() -> None:
    assert route_query("What are the main findings?").kind == "summary"
    table = route_query("What is the normal HbA1c value and reference range?")
    assert table.needs_table and table.needs_numeric
    comparison = route_query("Compare type 1 diabetes versus type 2 diabetes.")
    assert comparison.kind == "comparison"
    assert comparison.multi_hop


def test_document_aware_retrieval_self_corrects_weak_coverage() -> None:
    calls: list[str] = []

    def retrieve(query: str, top_k: int, where=None):
        calls.append(query)
        if len(calls) <= 8:
            return [Hit("d1", "Background unrelated passage about medicine.", {"document_id": "d1", "page_number": 2}, 0.03, vector_score=0.03, lexical_score=0.02)]
        return [Hit("d1", "Hyperthyroidism is associated with increased thyroid hormone production and clinical manifestations.", {"document_id": "d1", "page_number": 18, "section": "Clinical manifestations", "parent_id": "p18"}, 0.72, vector_score=0.72, lexical_score=0.80)]

    system = SimpleNamespace(retriever=SimpleNamespace(retrieve=retrieve), settings=SimpleNamespace(max_query_variants=8))
    hits, trace = retrieve_document_aware(system, "What are the clinical manifestations of hyperthyroidism?", top_k=4)
    assert hits
    assert trace["self_corrections"] == 1
    assert "clinical" in hits[0].text.lower()
    assert trace["coverage"]["overall"] > 0.0
    assert trace["final_hits"] <= 16


def test_generic_summary_does_not_get_fake_full_coverage_from_empty_query_terms() -> None:
    route = route_query("What are the main findings?")
    hits = [Hit("d", "A completely unrelated sentence about renal physiology.", {"document_id": "d", "page_number": 1}, 0.15)]
    coverage = evidence_coverage("What are the main findings?", route, hits)
    assert coverage["overall"] < 0.34
    assert coverage["sufficient"] is False


def test_summary_coverage_uses_document_structure() -> None:
    route = route_query("Summarize the main findings")
    hits = [
        Hit("d", "Introduction content.", {"document_id": "d", "page_number": 1, "section_id": "s1", "section": "Introduction"}, 0.70),
        Hit("d", "Clinical findings content.", {"document_id": "d", "page_number": 4, "section_id": "s2", "section": "Clinical findings"}, 0.75),
        Hit("d", "Diagnostic content.", {"document_id": "d", "page_number": 9, "section_id": "s3", "section": "Diagnosis"}, 0.80),
        Hit("d", "Treatment content.", {"document_id": "d", "page_number": 15, "section_id": "s4", "section": "Treatment"}, 0.78),
    ]
    coverage = evidence_coverage("Summarize the main findings", route, hits)
    assert coverage["basis"] == "document_structure_coverage"
    assert coverage["sufficient"] is True
    assert coverage["slots"]["overview"] > 0.34


def test_evidence_coverage_tracks_required_slots() -> None:
    route = route_query("Compare hypertension and diabetes mellitus")
    hits = [
        Hit("d", "Hypertension is elevated blood pressure.", {"page_number": 1}, 0.8),
        Hit("d", "Diabetes mellitus is a metabolic disorder affecting glucose regulation.", {"page_number": 3}, 0.8),
    ]
    coverage = evidence_coverage("Compare hypertension and diabetes mellitus", route, hits)
    assert coverage["entity_coverage"] > 0.0
    assert set(coverage["slots"]) == set(route.expected_slots)


def test_numeric_contradiction_detector_surfaces_disagreement() -> None:
    hits = [
        Hit("d", "The diagnostic threshold is 6.5%.", {"document_id": "d", "page_number": 10}, 0.8),
        Hit("d", "The diagnostic threshold is 7.0%.", {"document_id": "d", "page_number": 18}, 0.79),
    ]
    report = detect_contradictions(hits)
    assert report["has_contradiction"] is True


def test_authoritative_production_path_returns_a_certified_book_answer_without_llm() -> None:
    source = Hit(
        "endocrino",
        "Hyperthyroidism is associated with increased thyroid hormone production and can cause tachycardia.",
        {"document_id": "endocrino", "file_name": "endocrino.pdf", "page_number": 18, "page_numbers": [18], "section": "Clinical manifestations", "section_id": "s18", "chunk_id": "c18"},
        0.91,
        vector_score=0.91,
        lexical_score=0.95,
    )

    def retrieve(query: str, top_k: int, where=None):
        return [source]

    system = SimpleNamespace(
        retriever=SimpleNamespace(retrieve=retrieve),
        settings=SimpleNamespace(top_k=4),
        llm=None,
        citation_manager=SimpleNamespace(build=lambda hits: [], validate=lambda built, hits: []),
    )
    result = enhanced_god_answer(system, "What are the clinical manifestations of hyperthyroidism?")
    assert result["status"] in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    assert "Hyperthyroidism is associated with increased thyroid hormone production" in result["answer"]
    assert result["grounding"]["allow"] is True
    assert result["final_verification"]["allow"] is True
    assert result["generation_path"] == "deterministic_extractive"
    assert result["document_aware"] is True


def test_canonical_runtime_binds_authoritative_answer_function() -> None:
    from rag_project.app.production_rag import ProductionRAGSystem
    from rag_project.canonical_runtime import install
    from rag_project.intelligence import god_mode_100

    install()
    assert ProductionRAGSystem._certified_god_answer is god_mode_100.enhanced_god_answer
