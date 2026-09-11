from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from rag_project.intelligence.advanced_rag_engine import (
    detect_contradictions,
    evidence_coverage,
    retrieve_document_aware,
    route_query,
)


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
        if len(calls) == 1:
            return [Hit("d1", "Background unrelated passage about medicine.", {"document_id": "d1", "page_number": 2}, 0.15)]
        return [
            Hit(
                "d1",
                "Hyperthyroidism is associated with increased thyroid hormone production and clinical manifestations.",
                {"document_id": "d1", "page_number": 18, "section": "Clinical manifestations", "parent_id": "p18"},
                0.72,
            )
        ]

    system = SimpleNamespace(retriever=SimpleNamespace(retrieve=retrieve))
    hits, trace = retrieve_document_aware(system, "What are the clinical manifestations of hyperthyroidism?", top_k=4)
    assert hits
    assert trace["self_corrections"] == 1
    assert "clinical" in hits[0].text.lower()
    assert trace["coverage"]["overall"] > 0.0


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


def test_canonical_runtime_binds_authoritative_answer_function() -> None:
    from rag_project.app.production_rag import ProductionRAGSystem
    from rag_project.canonical_runtime import install
    from rag_project.intelligence import god_mode_100

    install()
    assert ProductionRAGSystem._certified_god_answer is god_mode_100.enhanced_god_answer
