from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

from rag_project.intelligence.advanced_reasoning import (
    abstention_ladder,
    build_parent_child_context,
    expand_neighbors,
    extract_evidence_structures,
    measurements_compatible,
    multi_hop_expand,
    normalize_numeric_measurements,
    parse_figure_evidence,
    parse_markdown_table,
    resolve_conflicts,
    sentence_compress,
)
from rag_project.intelligence.final_44 import FEATURE_IMPLEMENTATIONS, fail_closed, report, validate_citations
from rag_project.app.production_rag import ProductionRAGSystem


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    metadata: dict


def test_all_44_are_wired_to_concrete_implementations():
    assert len(FEATURE_IMPLEMENTATIONS) == 44
    assert len(set(FEATURE_IMPLEMENTATIONS)) == 44
    for target in FEATURE_IMPLEMENTATIONS.values():
        module_name, symbol = target.split(":", 1)
        assert callable(getattr(importlib.import_module(module_name), symbol))
    assert report()["all_features_wired"] is True


def test_structural_table_and_figure_evidence_is_real():
    table = parse_markdown_table("| Drug | Dose |\n| --- | --- |\n| A | 500 mg |\n| B | 1 g |")
    assert table is not None
    assert table.headers == ("Drug", "Dose")
    assert table.rows[0][1] == "500 mg"
    figures = parse_figure_evidence("Figure 3: Treatment response\nSee Figure 3 for the trend.")
    assert figures and figures[0].figure_id.lower().startswith("figure 3")
    structures = extract_evidence_structures("| A | 1 mg |\n| --- | --- |\n| B | 2 mg |\nFigure 1: chart")
    assert structures["table_ids"]
    assert structures["figure_ids"]


def test_numeric_units_convert_and_compare():
    values = normalize_numeric_measurements("Dose 1000 mg, volume 1 L, interval 2 h")
    assert any(v["unit"] == "mg" for v in values)
    mg = {"raw": "1000 mg", "value": 1000.0, "unit": "mg"}
    g = {"raw": "1 g", "value": 1.0, "unit": "g"}
    assert measurements_compatible(mg, g)


def test_parent_neighbor_multihop_and_compression():
    a = Hit("d1", "A", 0.9, {"chunk_id": "a", "document_id": "d1", "parent_id": "p", "page_numbers": [10]})
    b = Hit("d1", "B", 0.8, {"chunk_id": "b", "document_id": "d1", "parent_id": "p", "page_numbers": [10]})
    c = Hit("d1", "C", 0.7, {"chunk_id": "c", "document_id": "d1", "parent_id": "q", "page_numbers": [11]})
    parent = build_parent_child_context([a], [a, b, c])
    assert b in parent
    neighbors = expand_neighbors(parent, [a, b, c], radius=1)
    assert c in neighbors
    hops = multi_hop_expand("why", [a], [a, b, c], max_hops=2)
    assert any(h.hop == 2 for h in hops)
    compressed, state = sentence_compress("The dose is 500 mg. " * 10, "dose 500 mg", 60)
    assert compressed and state["compressed"] is True


def test_conflict_resolution_and_fail_closed_ladder():
    from rag_project.intelligence.evidence_guard import ClaimCheck

    claims = [
        ClaimCheck("Dose is 500 mg", 0.90, "SUPPORTED", ("S1",)),
        ClaimCheck("Dose is 600 mg", 0.55, "NUMERIC_MISMATCH", ("S2",), numeric_mismatch=True),
    ]
    result = resolve_conflicts(claims, {"S1": 0.9, "S2": 0.4})
    assert isinstance(result, dict)
    assert abstention_ladder(query_ok=True, retrieval_ok=True, evidence_score=0.9, grounding_ok=True, contradiction=False, generation_ok=True) == "ANSWER"
    assert fail_closed(query_ok=True, retrieval_ok=True, evidence_score=0.1, grounding_ok=True, contradiction=False, generation_ok=True)["allow"] is False


def test_citation_validation_rejects_nonexistent_sources():
    answer, state = validate_citations("Dose [S1] is okay [S99].", [Hit("d1", "dose", 1.0, {})])
    assert "[S1]" in answer
    assert "[S?]" in answer
    assert state["valid"] == [1]
    assert state["invalid"] == [99]


def test_production_answer_does_not_double_bind_wrapped_certifier(monkeypatch):
    system = ProductionRAGSystem.__new__(ProductionRAGSystem)
    system._production_feature_contract = {"all_resolved": True}
    system.settings = SimpleNamespace()

    def fake_certified_answer(self, question, metadata_filter=None):
        assert question == "What is the dose?"
        assert metadata_filter == {"document_id": "doc-1"}
        return {"status": "OK", "answer": "500 mg", "hits": [], "confidence": {}}

    monkeypatch.setattr(ProductionRAGSystem, "_certified_god_answer", fake_certified_answer)
    monkeypatch.setattr(
        "rag_project.app.production_rag.apply_medical_safety_policy",
        lambda question, result, settings: result,
    )
    monkeypatch.setattr(
        "rag_project.app.production_rag.sanitize_trace",
        lambda trace: trace,
    )

    result = system.answer("What is the dose?", {"document_id": "doc-1"})

    assert result["answer"] == "500 mg"
    assert result["production_contract"] == {"feature_count": 44, "all_features_resolved": True}
