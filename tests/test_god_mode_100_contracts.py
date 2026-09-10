from __future__ import annotations

from dataclasses import dataclass

from rag_project.intelligence.adaptive_retrieval import choose_retrieval_budget, should_retry_retrieval
from rag_project.intelligence.confidence_calibration import calibrate_confidence, confidence_gate
from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix, matrix_has_strong_support
from rag_project.intelligence.god_mode_100 import enhance_result
from rag_project.intelligence.hierarchical_evidence import build_evidence_hierarchy, select_context_levels


@dataclass
class Hit:
    doc_id: str
    text: str
    metadata: dict
    score: float = 0.8
    vector_score: float = 0.8
    lexical_score: float = 0.8


def _hit(text: str, chunk: str = "c1") -> Hit:
    return Hit("doc1", text, {"document_id": "doc1", "chunk_id": chunk, "page_numbers": [4], "section_title": "Treatment"})


def test_hierarchy_preserves_levels():
    records = build_evidence_hierarchy([_hit("Diabetes causes albuminuria. Next sentence.")])
    assert len(records) == 2
    assert records[0].document_id == "doc1"
    assert records[0].section_id == "Treatment"
    assert records[0].paragraph_id == "c1"
    assert records[0].sentence_id == "c1:s1"
    assert select_context_levels(records)["facts"] == 2


def test_entailment_matrix_finds_exact_supported_claim():
    matrix = build_claim_evidence_matrix(["Diabetes causes albuminuria."], [_hit("Diabetes causes albuminuria.")], ["S1"])
    assert matrix[0].support >= 0.70
    assert matrix[0].status == "ENTAILED"
    assert matrix_has_strong_support(matrix)
    assert matrix[0].evidence[0].start >= 0
    assert matrix[0].evidence[0].end > matrix[0].evidence[0].start


def test_adaptive_budget_escalates_weak_hard_query():
    budget = choose_retrieval_budget(query_tokens=30, entity_count=3, intent="mechanism", confidence=0.6, initial_score=0.2)
    assert budget.retry is True
    assert budget.depth == 3
    assert budget.rerank_limit >= 60
    assert should_retry_retrieval(alignment=0.2, entity_coverage=0.3, contradiction=0.0)


def test_confidence_calibration_is_conservative_under_conflict():
    confidence = calibrate_confidence(retrieval=0.9, rerank=0.9, entailment=0.8, entity_coverage=1.0, source_agreement=0.5, contradiction=1.0, safety_conflict=1.0)
    assert 0.0 <= confidence.calibrated <= 1.0
    assert confidence.calibrated < 0.8
    assert not confidence_gate(confidence, required=0.8)


def test_enhancer_adds_audit_layers_without_rewriting_answer():
    result = enhance_result(None, "What is diabetes?", {
        "status": "SUCCESS",
        "answer": "Diabetes is a metabolic disorder. [S1]",
        "hits": [_hit("Diabetes is a metabolic disorder.")],
        "claims": [{"claim": "Diabetes is a metabolic disorder.", "support": 0.9, "sources": ["S1"]}],
        "query_analysis": {"intent": "definition", "entities": ["diabetes mellitus"]},
        "semantic_understanding": {"confidence": 0.9},
        "semantic_alignment": {"score": 0.8, "entity_coverage": 1.0},
        "advanced_reasoning": {"entity_coverage": 1.0, "source_agreement": 0.5, "safety_conflict": 0.0},
        "grounding": {"supported_ratio": 0.9},
        "contradiction_report": {"has_contradiction": False},
        "small_model_assist": {"entities": ["diabetes"]},
    })
    assert result["god_mode_100"] is True
    assert result["answer"].startswith("Diabetes")
    assert result["evidence_claim_matrix"]
    assert result["evidence_hierarchy"]
    assert result["adaptive_retrieval_budget"]["depth"] >= 1
    assert result["confidence_calibration"]["calibrated"] >= 0.0
    assert result["validated_small_model_entities"] == ["diabetes"]
