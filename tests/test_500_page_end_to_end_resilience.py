from __future__ import annotations

from dataclasses import dataclass

from rag_project.intelligence.advanced_reasoning import full_reasoning_pass


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    metadata: dict


def test_500_page_reasoning_stays_bounded_and_cross_page_aware():
    hits = [
        Hit(
            "doc-500",
            f"Page {page}. Treatment dose is {500 + (page % 2)} mg. Figure {page % 7 + 1}: response.",
            1.0 - page / 5000,
            {
                "chunk_id": f"chunk-{page}",
                "document_id": "doc-500",
                "parent_id": f"section-{page // 10}",
                "section_id": f"section-{page // 10}",
                "page_numbers": [page],
                "page_quality": 0.95,
                "document_quality": 0.98,
            },
        )
        for page in range(1, 501)
    ]
    result = full_reasoning_pass("What dose is reported and which figure is referenced?", hits[:12], hits, max_context_chars=9000)
    assert result["route"]
    assert result["structures"]["figures"] >= 1
    assert result["hop_evidence"]
    assert len(result["compressed_context"]) <= 9000
    # The graph expansion is deliberately bounded rather than scanning all 500 pages into context.
    assert len(result["hop_evidence"]) <= 120
