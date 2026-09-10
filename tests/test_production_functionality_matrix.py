from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_project.application import ANSWER_PIPELINE_AUTHORITY, runtime_contract
from rag_project.canonical_runtime import ANSWER_AUTHORITY, CANONICAL_SERVICE, install as install_canonical_runtime
from rag_project.ingestion.ingestion_contract import (
    INGESTION_CONTRACT_VERSION,
    new_ingestion_request_id,
    wrap_ingest_callable,
)
from rag_project.intelligence import entity_coverage, final_answer_contract, god_mode_100, top_level_pipeline
from rag_project.intelligence.confidence_calibration import calibrate_confidence
from rag_project.intelligence.pipeline_integrity import (
    install as install_pipeline_integrity,
    is_control_message,
    safe_extract_query_entities,
    safe_rewrite_follow_up,
    safe_score_entity_coverage,
    safe_verify_final_answer,
)
from rag_project.intelligence.production_contract import (
    FEATURES,
    production_readiness,
    redact_sensitive_text,
    sanitize_trace,
    validate_feature_contract,
)
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
from rag_project.intelligence.query_intelligence import decompose_query, normalize_query, plan_query


@dataclass
class Hit:
    text: str
    score: float = 0.82
    doc_id: str = "doc-1"
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {
                "document_id": self.doc_id,
                "chunk_id": "chunk-1",
                "page_numbers": [1],
                "index_state": "READY",
            }


def hit(text: str, score: float = 0.82, page: int = 1) -> Hit:
    return Hit(
        text=text,
        score=score,
        metadata={
            "document_id": "doc-1",
            "chunk_id": f"chunk-{page}",
            "page_numbers": [page],
            "index_state": "READY",
        },
    )


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.calls: list[dict] = []

    def retrieve(self, query, top_k=8, where=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        return self.hits


class FakeMemory:
    def __init__(self, history=None):
        self.history = list(history or [])

    def prompt_context(self):
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history[-3:])


class FakeSystem:
    def __init__(self, hits=None, llm=None, history=None):
        self.retriever = FakeRetriever(hits)
        self.llm = llm
        self.conversation_memory = FakeMemory(history)
        self.settings = SimpleNamespace(
            top_k=8,
            temperature=0.2,
            context_token_budget=3200,
        )


@pytest.mark.parametrize(
    "query",
    [
        "What are the main findings?",
        "Summarize the document.",
        "Give the key findings.",
        "What does this document report?",
        "What was observed?",
        "List the principal observations.",
        "Quels sont les résultats principaux ?",
        "Résumez le document.",
        "Donnez les conclusions principales.",
        "ما هي أهم النتائج؟",
        "ما هي الاستنتاجات الرئيسية؟",
        "لخص الوثيقة.",
    ],
)
def test_summary_queries_have_no_fake_entities(query):
    assert safe_extract_query_entities(query) == ()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is diabetes mellitus?", "diabetes mellitus"),
        ("What is dapagliflozin?", "dapagliflozin"),
        ("What is metformin?", "metformin"),
        ("What is albuminuria?", "albuminuria"),
        ("What is hypertension?", "hypertension"),
        ("What is myocardial infarction?", "myocardial infarction"),
        ("What is CKD?", "ckd"),
        ("What is DKA?", "dka"),
        ("What is HbA1c?", "hba1c"),
    ],
)
def test_medical_entities_survive_normalization(query, expected):
    assert expected in safe_extract_query_entities(query)


@pytest.mark.parametrize(
    "label",
    [
        "Follow-up:",
        "Relevant entities:",
        "Query entities:",
        "Planned entities:",
        "Intent:",
        "Planner:",
        "Conversation context:",
        "Semantic understanding:",
        "Retrieval state:",
    ],
)
def test_internal_labels_never_pollute_standalone_query(label):
    result = safe_rewrite_follow_up("What are the main findings?", [])
    assert label.casefold() not in result.casefold()


@pytest.mark.parametrize(
    "query",
    [
        "What is diabetes?",
        "Define hypertension.",
        "What was observed?",
        "Résumez les résultats.",
        "ما هي النتائج الرئيسية؟",
    ],
)
def test_standalone_rewrite_is_idempotent(query):
    once = safe_rewrite_follow_up(query, [])
    twice = safe_rewrite_follow_up(once, [])
    assert once == twice


@pytest.mark.parametrize(
    "followup",
    [
        "What about this?",
        "What about that?",
        "And this?",
        "Et cela ?",
        "Et la suite ?",
        "وماذا عن ذلك؟",
        "هذا؟",
    ],
)
def test_followup_rewrite_uses_context_without_protocol_text(followup):
    rewritten = safe_rewrite_follow_up(
        followup,
        [("What are the complications of diabetes?", "Diabetic nephropathy is discussed.")],
    )
    assert rewritten
    assert "follow-up:" not in rewritten.casefold()
    assert "relevant entities:" not in rewritten.casefold()
    assert any(
        marker in rewritten.casefold()
        for marker in ("complications of diabetes", "diabetes mellitus", "diabetic nephropathy")
    )


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("What is diabetes?", "factual"),
        ("Define hypertension.", "factual"),
        ("What is the dose of metformin?", "numeric"),
        ("What causes cirrhosis?", "etiology"),
        ("How does insulin work?", "mechanism"),
        ("Compare diabetes and hypertension.", "comparison"),
    ],
)
def test_query_intent_classification(query, intent):
    assert plan_query(query).intent == intent


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  What   is   diabetes? ", "What is diabetes?"),
        ("what's diabetes?", "what is diabetes?"),
        ("Diabetes vs hypertension", "Diabetes versus hypertension"),
        ("A — B", "A - B"),
        ("A – B", "A - B"),
        ("  Arabic   text  ", "Arabic text"),
    ],
)
def test_query_normalization(raw, expected):
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    "query",
    [
        "dose and contraindications of metformin",
        "compare diabetes and hypertension",
        "causes and management of cirrhosis",
        "diagnosis and treatment of DKA",
        "what is the mechanism and prognosis of condition X",
    ],
)
def test_complex_queries_are_decomposed(query):
    pieces = decompose_query(query)
    assert len(pieces) >= 2
    assert all(piece.strip() for piece in pieces)


@pytest.mark.parametrize(
    "query",
    [
        "What are the main findings?",
        "What is diabetes?",
        "Define hypertension.",
        "Résumez le document.",
        "ما هي النتائج الرئيسية؟",
    ],
)
def test_simple_queries_do_not_become_multihop(query):
    plan = plan_query(query)
    assert plan.needs_multi_hop is False


@pytest.mark.parametrize(
    "query",
    [
        "What is the dose of metformin?",
        "What is the dosage of insulin?",
        "What percentage was reported?",
        "What is the value in mmol/L?",
        "What is 500 mg?",
    ],
)
def test_numeric_queries_enable_numeric_routing(query):
    assert plan_query(query).needs_numeric is True


@pytest.mark.parametrize(
    "query",
    [
        "What is in the table?",
        "Which row contains the value?",
        "Show the table entries.",
        "Quels sont les résultats du tableau ?",
        "أين يوجد الجدول؟",
    ],
)
def test_table_queries_enable_table_routing(query):
    assert plan_query(query).needs_table is True


@pytest.mark.parametrize(
    "query",
    [
        "What does Figure 1 show?",
        "Which diagram is referenced?",
        "Show the chart.",
        "Que montre la figure ?",
        "ماذا يوضح الشكل؟",
    ],
)
def test_figure_queries_enable_figure_routing(query):
    assert plan_query(query).needs_figure is True


@pytest.mark.parametrize(
    "query",
    [
        "What are the contraindications of metformin?",
        "How should DKA be managed?",
        "What causes cirrhosis?",
        "What is the mechanism of insulin?",
        "What is the prognosis?",
        "What is the relationship between diabetes and albuminuria?",
    ],
)
def test_clinical_reasoning_queries_are_complex(query):
    plan = plan_query(query)
    assert plan.needs_multi_hop or plan.intent in {"management", "etiology", "mechanism", "prognosis"}


@pytest.mark.parametrize(
    "message",
    [
        "The indexed evidence was insufficient to safely perform the required clinical synthesis.",
        "I could not verify a sufficiently grounded answer from the indexed evidence.",
        "I could not verify a sufficiently grounded answer from the indexed evidence; unsupported clinical details were withheld.",
        "The evidence was retrieved, but the required synthesis could not be verified without adding unsupported clinical content.",
    ],
)
def test_controlled_abstention_is_not_a_medical_claim(message):
    assert is_control_message(message)
    result = safe_verify_final_answer(message, [hit("Unrelated clinical evidence.")])
    assert result["checked"] is False
    assert result["allow"] is False
    assert result["claim_count"] == 0
    assert result["matrix_claim_count"] == 0


@pytest.mark.parametrize(
    ("answer", "evidence"),
    [
        ("Hyperglycemia was observed. [S1]", "Hyperglycemia was observed."),
        ("Albuminuria decreased. [S1]", "Albuminuria decreased."),
        ("The document reports diabetes. [S1]", "The document reports diabetes."),
    ],
)
def test_grounded_claims_are_verified(answer, evidence):
    result = safe_verify_final_answer(answer, [hit(evidence)])
    assert result["checked"] is True
    assert result["claim_count"] >= 1


def test_unrelated_claim_is_blocked():
    result = safe_verify_final_answer("Diabetes is present. [S1]", [hit("Hypertension is common.")])
    assert result["allow"] is False


def test_empty_answer_fails_closed():
    result = safe_verify_final_answer("", [], require_entailment=True)
    assert result["allow"] is False
    assert result["reason"] in {"no_verifiable_claims", "abstention_not_claim"}


@pytest.mark.parametrize(
    ("query", "evidence", "missing"),
    [
        ("What is diabetes mellitus?", "Diabetes mellitus is a metabolic disorder.", False),
        ("What is dapagliflozin?", "Dapagliflozin is discussed.", False),
        ("What is albuminuria?", "The text discusses albuminuria.", False),
        ("What is diabetes?", "The text discusses hypertension.", True),
    ],
)
def test_entity_coverage_reflects_evidence(query, evidence, missing):
    report = safe_score_entity_coverage(query, [hit(evidence)])
    assert bool(report["missing"]) is missing


@pytest.mark.parametrize(
    "query",
    [
        "What are the main findings?",
        "Summarize the document.",
        "Résumez les résultats.",
        "ما هي أهم النتائج؟",
    ],
)
def test_entity_free_coverage_has_no_missing_entities(query):
    report = safe_score_entity_coverage(query, [hit("Les résultats montrent une amélioration.")])
    assert report["entity_count"] == 0
    assert report["missing"] == []


def test_explicit_planner_labels_are_filtered():
    report = safe_score_entity_coverage(
        "What are the main findings?",
        [hit("The findings show improvement.")],
        planned_entities=("relevant entities", "main findings", "planner", "intent"),
    )
    assert all(item not in report["query_entities"] for item in ("relevant entities", "main findings", "planner", "intent"))


def test_evidence_provenance_is_preserved():
    report = safe_score_entity_coverage(
        "What is diabetes?",
        [hit("Diabetes is a metabolic disorder.", page=3), hit("Hypertension is common.", page=9)],
    )
    assert report["matches"]
    assert "S1" in report["matches"][0]["evidence_sources"]


@pytest.mark.parametrize(
    "text",
    [
        "contact user@example.com",
        "Call +213 555 123 456",
        "Patient ID 123456789",
        "record 987654321",
        "normal clinical text",
    ],
)
def test_trace_redaction_is_deterministic(text):
    assert redact_sensitive_text(text) == redact_sensitive_text(text)


def test_email_redaction():
    result = redact_sensitive_text("contact user@example.com")
    assert "user@example.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_phone_redaction():
    result = redact_sensitive_text("Call +213 555 123 456")
    assert "555 123 456" not in result
    assert "[REDACTED_PHONE]" in result


def test_long_id_redaction():
    result = redact_sensitive_text("patient 1234567890")
    assert "1234567890" not in result
    assert "[REDACTED_ID]" in result


def test_trace_sanitizer_does_not_mutate_input():
    source = {"question": "contact user@example.com", "score": 0.8}
    result = sanitize_trace(source)
    assert source["question"] == "contact user@example.com"
    assert result["question"] == "contact [REDACTED_EMAIL]"
    assert result["score"] == 0.8


@pytest.mark.parametrize(
    ("profile", "ready"),
    [
        ({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": True}, True),
        ({"feature_contract": False, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": True}, False),
        ({"feature_contract": True, "tests_green": False, "index_ready": True, "privacy_controls": True, "medical_safety": True}, False),
        ({"feature_contract": True, "tests_green": True, "index_ready": False, "privacy_controls": True, "medical_safety": True}, False),
        ({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": False, "medical_safety": True}, False),
        ({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": False}, False),
    ],
)
def test_release_readiness_requires_every_gate(profile, ready):
    assert production_readiness(profile)["release_ready"] is ready


def test_feature_contract_is_complete_and_resolvable():
    report = validate_feature_contract()
    assert report["feature_count"] == 44
    assert report["unique_names"] is True
    assert report["duplicates"] == []
    assert report["unresolved"] == {}
    assert report["all_resolved"] is True


def test_runtime_contract_declares_single_authority():
    contract = runtime_contract()
    assert contract["composition_root"] == "rag_project.application.create_rag_system"
    assert contract["canonical_service"] == CANONICAL_SERVICE
    assert contract["canonical_ingestion"].endswith("robust_ingest_file")
    assert contract["answer_pipeline_authority"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_pipeline_execution"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_monkey_patch"] is False
    assert contract["structured_request_context"] is True
    assert contract["structured_evidence_bundle"] is True
    assert contract["structured_answer_envelope"] is True
    assert contract["confidence_breakdown"] is True
    assert contract["request_traceability"] is True
    assert contract["ingestion_traceability"] is True
    assert contract["atomic_ingestion_publication"] is True
    assert contract["post_write_index_validation"] is True
    assert contract["production_contract_version"] == CONTRACT_VERSION
    assert contract["ingestion_contract_version"] == INGESTION_CONTRACT_VERSION


def test_canonical_runtime_binding_is_installed():
    result = install_canonical_runtime()
    assert result["canonical_service"] == CANONICAL_SERVICE
    assert result["answer_pipeline_authority"] == ANSWER_AUTHORITY
    assert result["class_binding_installed"] is True
    assert result["runtime_contract_bound"] is True


def test_pipeline_integrity_install_is_idempotent():
    install_pipeline_integrity()
    first = top_level_pipeline.rewrite_follow_up
    install_pipeline_integrity()
    assert top_level_pipeline.rewrite_follow_up is first
    assert entity_coverage.extract_query_entities is safe_extract_query_entities
    assert final_answer_contract.verify_final_answer is safe_verify_final_answer
    assert god_mode_100.verify_final_answer is safe_verify_final_answer


def test_ingestion_request_id_format():
    assert re.fullmatch(r"ing-[0-9a-f]{16}", new_ingestion_request_id())


def test_ingestion_request_ids_are_unique():
    assert len({new_ingestion_request_id() for _ in range(128)}) == 128


def test_ingestion_wrapper_adds_trace_without_changing_payload():
    def original(system, pdf_path, *args, **kwargs):
        return {"status": "READY", "document_id": "doc-1", "count": 4}

    result = wrap_ingest_callable(original)(object(), Path("doc.pdf"))
    assert result["status"] == "READY"
    assert result["document_id"] == "doc-1"
    assert result["count"] == 4
    assert re.fullmatch(r"ing-[0-9a-f]{16}", result["ingestion_request_id"])
    assert result["ingestion_trace_version"] == INGESTION_CONTRACT_VERSION
    assert result["ingestion_elapsed_ms"] >= 0
    assert result["ingestion_contract"]["atomic_publication"] is True
    assert result["ingestion_contract"]["lease_fencing"] is True
    assert result["ingestion_contract"]["post_write_validation"] is True


def test_ingestion_wrapper_preserves_non_dict_result():
    def original(system, pdf_path):
        return True

    assert wrap_ingest_callable(original)(object(), "doc.pdf") is True


def test_ingestion_wrapper_is_idempotent():
    def original(system, pdf_path):
        return {"ok": True}

    first = wrap_ingest_callable(original)
    assert wrap_ingest_callable(first) is first
    assert wrap_ingest_callable(first) is first


def test_ingestion_wrapper_reraises_original_exception():
    def original(system, pdf_path):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        wrap_ingest_callable(original)(SimpleNamespace(logger=None), "doc.pdf")


@pytest.mark.parametrize("bad_path", ["missing.pdf", "document.txt", "scan.docx", "data.csv", "image.png"])
def test_robust_ingestor_rejects_missing_or_wrong_extension(bad_path):
    with pytest.raises(ValueError, match="Unsupported or missing PDF"):
        from rag_project.ingestion.robust_ingestor import robust_ingest_file

        robust_ingest_file(SimpleNamespace(), bad_path)


def test_robust_ingestor_rejects_pdf_directory():
    from rag_project.ingestion.robust_ingestor import robust_ingest_file

    with pytest.raises(ValueError, match="Unsupported or missing PDF"):
        with pytest.MonkeyPatch.context() as patch:
            # os-level temp directory through pathlib keeps this test dependency-free.
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "folder.pdf"
                directory.mkdir()
                robust_ingest_file(SimpleNamespace(), directory)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_confidence_calibration_stays_in_bounds(value):
    result = calibrate_confidence(
        retrieval=value,
        rerank=value,
        entailment=value,
        entity_coverage=value,
        source_agreement=value,
        contradiction=0.0,
        safety_conflict=0.0,
    )
    assert 0.0 <= result.calibrated <= 1.0
    assert result.level in {"low", "medium", "high"}
    assert all(0.0 <= factor <= 1.0 for factor in result.factors.values())


def test_confidence_penalizes_contradiction():
    clean = calibrate_confidence(
        retrieval=0.9,
        rerank=0.9,
        entailment=0.9,
        entity_coverage=1.0,
        source_agreement=0.9,
        contradiction=0.0,
        safety_conflict=0.0,
    )
    conflicted = calibrate_confidence(
        retrieval=0.9,
        rerank=0.9,
        entailment=0.9,
        entity_coverage=1.0,
        source_agreement=0.9,
        contradiction=1.0,
        safety_conflict=0.0,
    )
    assert conflicted.calibrated < clean.calibrated


def test_confidence_penalizes_safety_conflict():
    clean = calibrate_confidence(
        retrieval=0.9,
        rerank=0.9,
        entailment=0.9,
        entity_coverage=1.0,
        source_agreement=0.9,
        contradiction=0.0,
        safety_conflict=0.0,
    )
    risky = calibrate_confidence(
        retrieval=0.9,
        rerank=0.9,
        entailment=0.9,
        entity_coverage=1.0,
        source_agreement=0.9,
        contradiction=0.0,
        safety_conflict=1.0,
    )
    assert risky.calibrated < clean.calibrated


def test_confidence_clamps_out_of_range_inputs():
    result = calibrate_confidence(
        retrieval=-1.0,
        rerank=2.0,
        entailment=-5.0,
        entity_coverage=3.0,
        source_agreement=-2.0,
        contradiction=4.0,
        safety_conflict=9.0,
    )
    assert all(0.0 <= factor <= 1.0 for factor in result.factors.values())
    assert 0.0 <= result.calibrated <= 1.0


def test_top_level_simple_query_is_clean():
    system = FakeSystem(hits=[hit("The main findings include hyperglycemia and ketoacidosis.")])
    result = top_level_pipeline.complete_phases(
        system,
        "What are the main findings?",
        {"hits": system.retriever.hits},
    )
    assert result["rewritten_question"] == "What are the main findings?"
    assert result["phase_plan"]["entities"] == ()
    assert result["phase_plan"]["needs_multi_hop"] is False
    assert result["extractive_stage"]["supported"] is True
    assert result["answer"]
    assert result.get("status") != "GENERATION_ABSTAIN"


def test_top_level_hard_query_escalates_retrieval():
    system = FakeSystem(hits=[hit("Metformin dose information: 500 mg.")])
    result = top_level_pipeline.complete_phases(system, "What is the dose of metformin?", {"hits": []})
    assert result["adaptive_retrieval"]["escalated"] is True
    assert system.retriever.calls
    assert len(system.retriever.calls) <= 8


def test_top_level_output_contains_phase_visibility():
    system = FakeSystem(hits=[hit("Hyperglycemia was reported.")])
    result = top_level_pipeline.complete_phases(system, "What was reported?", {"hits": system.retriever.hits})
    for key in (
        "phase_plan",
        "adaptive_retrieval",
        "context_compression",
        "extractive_stage",
        "two_stage_policy",
        "phases",
    ):
        assert key in result
    assert set(result["phases"]) >= {
        "phase_1_query_understanding",
        "phase_2_retrieval_precision",
        "phase_3_two_stage_generation",
        "phase_4_verification",
        "phase_5_intelligence_visibility",
    }


def test_top_level_extractive_answer_keeps_source_marker():
    system = FakeSystem(hits=[hit("Hyperglycemia was observed.")])
    result = top_level_pipeline.complete_phases(system, "What was observed?", {"hits": system.retriever.hits})
    assert "[S1]" in result["answer"] or "[S1]" in result["extractive_stage"]["draft"]


@pytest.mark.parametrize(
    "query",
    [
        "What is diabetes?",
        "What is dapagliflozin?",
        "What is albuminuria?",
        "What is CKD?",
        "What is DKA?",
    ],
)
def test_medical_query_plans_have_semantic_signal(query):
    plan = plan_query(query)
    assert plan.entities or plan.intent in {"factual", "definition"}
    assert plan.variants
    assert plan.subqueries


def test_runtime_contract_is_json_serializable():
    json.dumps(runtime_contract())


def test_production_readiness_is_json_serializable():
    json.dumps(
        production_readiness(
            {
                "feature_contract": True,
                "tests_green": True,
                "index_ready": True,
                "privacy_controls": True,
                "medical_safety": True,
            }
        )
    )


def test_ingestion_trace_result_is_json_serializable():
    def original(system, pdf_path):
        return {"status": "READY", "count": 1}

    result = wrap_ingest_callable(original)(object(), "document.pdf")
    json.dumps(result)


def test_entity_report_is_json_serializable():
    json.dumps(safe_score_entity_coverage("What is diabetes?", [hit("Diabetes is a metabolic disorder.")]), default=str)


def test_final_verification_is_json_serializable():
    json.dumps(safe_verify_final_answer("Hyperglycemia was observed. [S1]", [hit("Hyperglycemia was observed.")]), default=str)


def test_protocol_text_never_appears_in_clean_pipeline_query():
    system = FakeSystem(hits=[hit("The document reports improved outcomes.")])
    result = top_level_pipeline.complete_phases(system, "What are the main findings?", {"hits": system.retriever.hits})
    query = result["rewritten_question"].casefold()
    assert "follow-up:" not in query
    assert "relevant entities:" not in query
    assert "query entities:" not in query


def test_entity_coverage_has_structured_fields():
    result = safe_score_entity_coverage("What is diabetes?", [hit("Diabetes is a metabolic disorder.")])
    for field in (
        "query_entities",
        "entity_count",
        "covered",
        "missing",
        "partial",
        "coverage",
        "partial_coverage",
        "per_entity",
        "matches",
    ):
        assert field in result


def test_final_verification_has_structured_fields():
    result = safe_verify_final_answer("Hyperglycemia was observed. [S1]", [hit("Hyperglycemia was observed.")])
    for field in (
        "checked",
        "allow",
        "reason",
        "claim_count",
        "blocked_claims",
        "supported_ratio",
        "matrix_claim_count",
        "matrix_all_entailed",
        "claim_checks",
        "evidence_claim_matrix",
    ):
        assert field in result


@pytest.mark.parametrize("query", [
    "What is diabetes?",
    "What is dapagliflozin?",
    "What are the complications of diabetes?",
    "What is the dose of metformin?",
    "How should DKA be managed?",
])
def test_query_plans_are_bounded(query):
    plan = plan_query(query)
    assert len(plan.subqueries) <= 8
    assert len(plan.variants) <= 10
    assert len(plan.entities) <= 16


def test_canonical_constants_match():
    assert ANSWER_AUTHORITY == ANSWER_PIPELINE_AUTHORITY
    assert CANONICAL_SERVICE == "rag_project.app.production_rag.ProductionRAGSystem"


def test_release_gate_requires_medical_safety():
    profile = {
        "feature_contract": True,
        "tests_green": True,
        "index_ready": True,
        "privacy_controls": True,
        "medical_safety": False,
    }
    assert production_readiness(profile)["release_ready"] is False


def test_release_gate_requires_privacy():
    profile = {
        "feature_contract": True,
        "tests_green": True,
        "index_ready": True,
        "privacy_controls": False,
        "medical_safety": True,
    }
    assert production_readiness(profile)["release_ready"] is False


def test_index_state_contract_is_present_in_fixture():
    evidence = hit("Diabetes is a metabolic disorder.")
    assert evidence.metadata["index_state"] == "READY"


@pytest.mark.parametrize("score", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_hit_scores_are_accepted_as_valid_retrieval_inputs(score):
    evidence = hit("Clinical evidence.", score=score)
    assert 0.0 <= evidence.score <= 1.0


def test_control_message_with_source_marker_is_not_misclassified_when_medical_content_is_present():
    assert is_control_message("Hyperglycemia was observed. [S1]") is False


def test_safe_rewrite_handles_empty_history():
    assert safe_rewrite_follow_up("What about this?", []) == "What about this?"


def test_safe_rewrite_handles_none_history():
    assert safe_rewrite_follow_up("What about this?", None) == "What about this?"


def test_safe_rewrite_handles_empty_question():
    assert safe_rewrite_follow_up("", [("What is diabetes?", "Diabetes is a condition.")]) == ""


def test_safe_entity_layer_rejects_plain_noun_phrases():
    phrases = [
        "main findings",
        "relevant entities",
        "query context",
        "conversation history",
        "final answer",
        "retrieval state",
    ]
    for phrase in phrases:
        assert safe_extract_query_entities(phrase) == ()


def test_installations_keep_the_runtime_contract_available():
    install_pipeline_integrity()
    install_canonical_runtime()
    contract = runtime_contract()
    assert contract["answer_pipeline_execution"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["ingestion_traceability"] is True
