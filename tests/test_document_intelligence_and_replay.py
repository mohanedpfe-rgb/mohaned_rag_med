from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_project.intelligence.document_intelligence import build_document_map, section_coverage
from rag_project.intelligence.medical_knowledge_graph import build_graph, expand_from_query
from rag_project.intelligence.retrieval_replay import failure_summary, load, record


@dataclass
class Hit:
    doc_id: str
    text: str
    metadata: dict


def test_document_map_builds_chapter_and_section_structure() -> None:
    hits = [
        Hit("book", "Introduction text", {"document_id": "book", "page_number": 1, "chapter": "Chapter 1", "section": "Introduction", "section_id": "s1", "chunk_id": "c1"}),
        Hit("book", "Diagnosis text", {"document_id": "book", "page_number": 12, "chapter": "Chapter 1", "section": "Diagnosis", "section_id": "s2", "chunk_id": "c2"}),
    ]
    result = build_document_map(hits)
    assert result["document_count"] == 1
    assert result["node_count"] >= 3
    assert section_coverage(hits)["distinct_sections"] == 2


def test_document_derived_graph_expands_multi_hop_queries() -> None:
    hits = [
        Hit("book", "Graves disease is associated with hyperthyroidism. Hyperthyroidism causes tachycardia.", {"entities": ["Graves disease", "hyperthyroidism", "tachycardia"]}),
    ]
    graph = build_graph(hits)
    assert graph["node_count"] >= 2
    expanded = expand_from_query("What causes hyperthyroidism?", graph)
    assert expanded
    assert "tachycardia" in expanded[0].lower() or "graves disease" in expanded[0].lower()


def test_replay_is_bounded_and_failure_summary_is_actionable(tmp_path: Path) -> None:
    record(tmp_path, {"success": True, "status": "SUCCESS", "routing": {"kind": "factual"}})
    record(tmp_path, {"success": False, "status": "ANSWER_UNAVAILABLE", "failure_stage": "verification", "routing": {"kind": "summary"}})
    rows = load(tmp_path)
    assert len(rows) == 2
    summary = failure_summary(tmp_path)
    assert summary["failure_count"] == 1
    assert summary["by_stage"]["verification"] == 1
