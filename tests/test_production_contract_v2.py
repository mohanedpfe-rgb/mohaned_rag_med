from types import SimpleNamespace

from rag_project.intelligence import god_mode_100
from rag_project.intelligence.production_contract_v2 import (
    apply_contract,
    build_evidence_bundle,
    build_request_context,
    compute_confidence,
    install,
)
from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION, wrap_ingest_callable


def _hit(text: str, score: float = 0.82):
    return SimpleNamespace(
        text=text,
        score=score,
        doc_id="doc-1",
        metadata={"document_id": "doc-1", "file_name": "guide.pdf", "page_numbers": [4, 5]},
    )


def test_short_standalone_question_is_not_inherited_from_history():
    context = build_request_context(
        "What is sepsis?",
        history=(("What are the complications of diabetes?", "The guide describes diabetic nephropathy."),),
    )
    assert context.is_followup is False
    assert context.conversation_used is False
    assert context.canonical_question == "What is sepsis?"
    assert context.complexity == "simple"


def test_true_followup_uses_context_without_protocol_text():
    context = build_request_context(
        "What about this?",
        history=(("What are the complications of diabetes?", "The guide describes diabetic nephropathy."),),
    )
    assert context.is_followup is True
    assert context.conversation_used is True
    assert "follow-up:" not in context.canonical_question.casefold()
    assert "relevant entities:" not in context.canonical_question.casefold()


def test_request_context_contains_stable_request_metadata():
    context = build_request_context("What are the main findings?", request_id="rag-test-123")
    assert context.request_id == "rag-test-123"
    assert context.intent == "factual"
    assert context.entities == ()
    assert context.needs_multi_hop is False


def test_evidence_bundle_preserves_source_provenance():
    bundle = build_evidence_bundle("rag-test-123", [_hit("Hyperglycemia was observed.")], entity_coverage={"coverage": 1.0})
    assert bundle.hit_count == 1
    assert bundle.top_score == 0.82
    assert bundle.sources[0].source_id == "S1"
    assert bundle.sources[0].file_name == "guide.pdf"
    assert bundle.sources[0].pages == (4, 5)


def test_confidence_failure_does_not_create_synthetic_contradiction():
    confidence = compute_confidence(
        retrieval=0.9,
        evidence_quality=0.8,
        entailment=0.0,
        entity_coverage=1.0,
        verification=0.0,
        contradiction=0.0,
        hard_gate_passed=False,
    )
    assert confidence.contradiction == 0.0
    assert confidence.final <= 0.49


def test_apply_contract_represents_abstention_as_state():
    result = apply_contract(
        {
            "status": "REASONING_ABSTAIN",
            "abstained": True,
            "answer": "I could not verify a sufficiently grounded answer from the indexed evidence; unsupported or conflicting clinical details were withheld.",
            "hits": [_hit("Unrelated evidence.")],
            "entity_coverage": {"coverage": 0.0, "missing": []},
            "final_verification": {"allow": False, "reason": "blocked_claims", "supported_ratio": 0.0, "blocked_claims": 0, "claim_checks": []},
        },
        build_request_context("What are the main findings?", request_id="rag-test-124"),
    )
    assert result["contract_status"] == "abstain"
    assert result["answer_envelope"]["status"] == "abstain"
    assert result["confidence_breakdown"]["contradiction"] == 0.0
    assert result["answer_envelope"]["request_id"] == "rag-test-124"


def test_ingestion_wrapper_adds_trace_without_changing_success_payload():
    def original(system, pdf_path):
        return {"status": "ready", "file_name": str(pdf_path)}

    wrapped = wrap_ingest_callable(original)
    result = wrapped(object(), "guide.pdf")
    assert result["status"] == "ready"
    assert result["file_name"] == "guide.pdf"
    assert result["ingestion_trace_version"] == INGESTION_CONTRACT_VERSION
    assert result["ingestion_request_id"].startswith("ing-")
    assert result["ingestion_contract"]["atomic_publication"] is True


def test_contract_install_patches_canonical_enhancer_once():
    install()
    first = god_mode_100.enhance_result
    install()
    assert god_mode_100.enhance_result is first
