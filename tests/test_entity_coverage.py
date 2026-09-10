from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.entity_coverage import extract_query_entities, score_entity_coverage


def hit(text: str):
    return SimpleNamespace(text=text, metadata={"document_id": "d1", "chunk_id": "c1"}, doc_id="d1")


def test_query_entity_extraction_keeps_open_set_terms() -> None:
    entities = extract_query_entities("What is the relationship between dapagliflozin and albuminuria?")
    assert any("dapagliflozin" in entity for entity in entities)
    assert any("albuminuria" in entity for entity in entities)


def test_entity_coverage_reports_missing_entities() -> None:
    result = score_entity_coverage(
        "What is the relationship between dapagliflozin and albuminuria?",
        [hit("Albuminuria was reduced in participants treated with dapagliflozin.")],
    )
    assert result["entity_count"] >= 2
    assert result["coverage"] >= 0.5
    assert not result["missing"]


def test_entity_coverage_does_not_hide_absent_entity() -> None:
    result = score_entity_coverage(
        "What is the relationship between dapagliflozin and albuminuria?",
        [hit("Albuminuria was reduced in participants.")],
    )
    assert result["missing"]
    assert result["coverage"] < 1.0
