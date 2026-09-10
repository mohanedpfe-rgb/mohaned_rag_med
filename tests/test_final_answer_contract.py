from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.final_answer_contract import verify_final_answer


def hit(text: str, *, chunk_id: str = "c1"):
    return SimpleNamespace(
        text=text,
        doc_id="d1",
        score=0.9,
        vector_score=0.8,
        lexical_score=0.7,
        metadata={"document_id": "d1", "chunk_id": chunk_id, "page_numbers": [1]},
    )


def test_final_answer_contract_allows_supported_answer() -> None:
    evidence = [hit("Metformin is associated with improved glycemic control in type 2 diabetes.")]
    result = verify_final_answer(
        "Metformin is associated with improved glycemic control in type 2 diabetes. [S1]",
        evidence,
    )
    assert result["checked"] is True
    assert result["allow"] is True
    assert result["blocked_claims"] == 0
    assert result["supported_ratio"] >= 0.60


def test_final_answer_contract_blocks_unsupported_claim() -> None:
    evidence = [hit("Metformin is associated with improved glycemic control in type 2 diabetes.")]
    result = verify_final_answer("Metformin cures cancer. [S1]", evidence)
    assert result["checked"] is True
    assert result["allow"] is False
    assert result["blocked_claims"] >= 1
    assert result["reason"] in {"blocked_claims", "support_ratio_below_threshold"}


def test_final_answer_contract_blocks_numeric_mismatch() -> None:
    evidence = [hit("The dose was 500 mg twice daily.")]
    result = verify_final_answer("The dose was 1000 mg twice daily. [S1]", evidence, require_entailment=True)
    assert result["allow"] is False
    assert result["blocked_claims"] >= 1


def test_final_answer_contract_requires_entailment_for_hard_queries() -> None:
    evidence = [hit("ACE inhibitors are associated with reduced albuminuria.")]
    result = verify_final_answer("ACE inhibitors are associated with reduced albuminuria. [S1]", evidence, require_entailment=True)
    assert result["allow"] is True
    assert result["matrix_all_entailed"] is True
