from __future__ import annotations

from rag_project.intelligence.advanced_clinical_reasoner import (
    assess_clinical_reasoning,
    extract_clinical_facts,
    build_reasoning_instruction,
)
from rag_project.intelligence.semantic_reasoning import understand_query
from rag_project.retrieval.hybrid_retriever import RetrievalHit


def _hit(text: str, *, doc: str = "d1", chunk: str = "c1", score: float = 0.8):
    return RetrievalHit(
        doc_id=doc,
        text=text,
        metadata={"document_id": doc, "chunk_id": chunk, "page_numbers": [1]},
        score=score,
        vector_score=score,
        lexical_score=0.4,
    )


def test_extracts_explicit_causal_fact():
    facts = extract_clinical_facts("Diabetes causes diabetic nephropathy.")
    assert facts
    assert facts[0].subject == "diabetes mellitus"
    assert facts[0].object == "diabetic nephropathy"
    assert facts[0].predicate == "causes"
    assert facts[0].polarity == 1


def test_extracts_explicit_management_fact():
    facts = extract_clinical_facts("Hypertension is treated with insulin.")
    assert facts
    assert facts[0].predicate == "treated_with"


def test_negative_causal_statement_is_not_positive_support():
    facts = extract_clinical_facts("Diabetes does not cause diabetic nephropathy.")
    assert facts
    assert facts[0].polarity == -1


def test_two_step_path_is_found():
    understanding = understand_query("Why does diabetes cause albuminuria?")
    hits = [
        _hit("Diabetes causes diabetic nephropathy.", doc="d1", chunk="c1"),
        _hit("Diabetic nephropathy is associated with albuminuria.", doc="d2", chunk="c2"),
    ]
    result = assess_clinical_reasoning(understanding, hits)
    assert result.mode in {"MULTI_HOP", "ONE_HOP"}
    assert result.depth >= 1
    assert result.path_support > 0
    assert result.entity_coverage >= 0.5


def test_unconnected_entities_are_not_promoted_to_a_reasoning_chain():
    understanding = understand_query("Why does diabetes cause albuminuria?")
    hits = [
        _hit("Diabetes is common.", chunk="c1"),
        _hit("Albuminuria can occur in kidney disease.", chunk="c2"),
    ]
    result = assess_clinical_reasoning(understanding, hits)
    assert result.mode == "INSUFFICIENT"
    assert result.allow_generation is False
    assert "no_explicit_reasoning_path" in result.blocked_reasons


def test_conflicting_same_claim_blocks_generation():
    understanding = understand_query("Is diabetes associated with albuminuria?")
    hits = [
        _hit("Diabetes is associated with albuminuria.", doc="d1", chunk="c1"),
        _hit("Diabetes is not associated with albuminuria.", doc="d2", chunk="c2"),
    ]
    result = assess_clinical_reasoning(understanding, hits)
    assert result.contradiction > 0
    assert result.allow_generation is False
    assert "conflicting_evidence" in result.blocked_reasons


def test_safety_conflict_is_blocked():
    understanding = understand_query("What is the treatment and contraindications for hypertension?")
    hits = [
        _hit("Hypertension is treated with insulin.", chunk="c1"),
        _hit("Insulin is contraindicated in hypertension.", chunk="c2"),
    ]
    result = assess_clinical_reasoning(understanding, hits)
    assert result.safety_conflict > 0
    assert result.allow_generation is False


def test_direct_answer_can_pass_without_fake_multihop_reasoning():
    understanding = understand_query("What is diabetes?")
    hits = [_hit("Diabetes mellitus is a chronic metabolic disease.")]
    result = assess_clinical_reasoning(understanding, hits)
    assert result.mode == "DIRECT"
    assert result.allow_generation is True
    assert result.depth == 1


def test_reasoning_instruction_is_depth_sensitive():
    understanding = understand_query("Why does diabetes cause albuminuria?")
    hits = [
        _hit("Diabetes causes diabetic nephropathy.", doc="d1", chunk="c1"),
        _hit("Diabetic nephropathy is associated with albuminuria.", doc="d2", chunk="c2"),
    ]
    result = assess_clinical_reasoning(understanding, hits)
    instruction = build_reasoning_instruction(understanding, result)
    assert "explicit evidence path" in instruction.lower()
    assert "do not invent" in instruction.lower()
