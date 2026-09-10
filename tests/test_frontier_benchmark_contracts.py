from __future__ import annotations

from dataclasses import dataclass

from rag_project.evaluation.frontier_benchmark import BenchmarkCase, run_benchmark


@dataclass
class Hit:
    doc_id: str
    metadata: dict
    text: str = ""


def test_benchmark_reports_core_metrics_and_categories():
    cases = [
        BenchmarkCase("d1", "definition", "What is diabetes?", ("diabetes mellitus",), "definition", ("c1",), True),
        BenchmarkCase("u1", "unanswerable", "What is unicorn disease?", (), "factual", (), False),
    ]

    def runner(question: str) -> dict:
        if "diabetes" in question:
            return {
                "status": "SUCCESS",
                "query_analysis": {"entities": ["diabetes mellitus"], "intent": "definition"},
                "hits": [Hit("doc", {"chunk_id": "c1"})],
                "grounding": {"allow": True},
                "claims": [{"support": 0.9, "sources": ["S1"]}],
            }
        return {"status": "NOT_SUPPORTED", "query_analysis": {"entities": [], "intent": "factual"}, "hits": [], "grounding": {"allow": False}, "claims": []}

    report = run_benchmark(cases, runner)
    assert report.total == 2
    assert report.entity_recall == 0.5
    assert report.intent_accuracy == 1.0
    assert report.evidence_recall == 0.5
    assert report.abstention_accuracy == 1.0
    assert "definition" in report.by_category
    assert "unanswerable" in report.by_category


def test_benchmark_handles_empty_suite():
    report = run_benchmark([], lambda _: {})
    assert report.total == 0
    assert report.entity_recall == 0.0
