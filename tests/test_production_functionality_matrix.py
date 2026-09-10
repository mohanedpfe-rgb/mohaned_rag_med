from __future__ import annotations

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
from rag_project.intelligence.production_contract import (
    FEATURES,
    redact_sensitive_text,
    production_readiness,
    sanitize_trace,
    validate_feature_contract,
)
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
from rag_project.intelligence.query_intelligence import (
    classify_intent,
    decompose_query,
    normalize_query,
    plan_query,
)
from rag_project.intelligence.pipeline_integrity import (
    install as install_pipeline_integrity,
    is_control_message,
    safe_extract_query_entities,
    safe_rewrite_follow_up,
    safe_score_entity_coverage,
    safe_verify_final_answer,
)


@dataclass
class Hit:
    text: str
    score: float = 0.82
    doc_id: str = "doc-1"
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {"document_id": self.doc_id, "chunk_id": "chunk-1", "page_numbers": [1]}


def hit(text: str, score: float = 0.82, page: int = 1) -> Hit:
    return Hit(text=text, score=score, metadata={"document_id": "doc-1", "chunk_id": f"chunk-{page}", "page_numbers": [page]})


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.calls: list[dict] = []

    def retrieve(self, query, top_k=8, where=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        return self.hits


class FakeLLM:
    def __init__(self, text: str = ""):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.text


class FakeMemory:
    def __init__(self, history=None):
        self.history = list(history or [])

    def prompt_context(self):
        return " ".join(q for q, _ in self.history[-3:])


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
    "text",
    [
        "What are the main findings?",
        "Summarize the document.",
        "Give the key findings.",
        "What does this document report?",
        "Résume les résultats principaux.",
        "Quels sont les résultats principaux ?",
        "ما هي أهم النتائج؟",
        "Donnez un résumé des résultats.",
        "List the principal observations.",
        "What was observed?",
    ],
)
def test_entity_free_queries_never_gain_generic_entities(text):
    entities = safe_extract_query_entities(text)
    assert entities == ()


@pytest.mark.parametrize(
    "entity_query,expected",
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
        ("What is 500 mg?", "500 mg"),
    ],
)
def test_real_medical_entities_are_retained(entity_query, expected):
    entities = safe_extract_query_entities(entity_query)
    assert expected in entities


@pytest.mark.parametrize("label", [
    "Follow-up:",
    "Relevant entities:",
    "Query entities:",
    "Planner:",
    "Intent:",
    "Conversation context:",
    "Semantic understanding:",
    "Retrieval state:",
])
def test_protocol_labels_never_enter_standalone_rewrite(label):
    result = safe_rewrite_follow_up("What are the main findings?", [])
    assert label.casefold() not in result.casefold()


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What are the complications of diabetes?",
        "What is dapagliflozin?",
        "What is albuminuria?",
        "Define hypertension.",
        "Qu'est-ce que le diabète ?",
        "ما هو السكري؟",
    ],
)
def test_standalone_rewrite_is_idempotent(question):
    once = safe_rewrite_follow_up(question, [])
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
        "هذا؟",
        "وماذا عن ذلك؟",
    ],
)
def test_real_followups_use_context_without_internal_metadata(followup):
    result = safe_rewrite_follow_up(
        followup,
        [("What are the complications of diabetes?", "Diabetic nephropathy is discussed.")],
    )
    assert result
    assert "follow-up:" not in result.casefold()
    assert "relevant entities:" not in result.casefold()
    assert "complications of diabetes" in result.casefold() or "diabetes mellitus" in result.casefold() or "diabetic nephropathy" in result.casefold()


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("What is diabetes?", "factual"),
        ("Define hypertension.", "factual"),
        ("What is the dose of metformin?", "numeric"),
        ("What are the contraindications of metformin?", "management"),
        ("What causes cirrhosis?", "etiology"),
        ("How does insulin work?", "mechanism"),
        ("Compare diabetes and hypertension.", "comparison"),
    ],
)
def test_query_planner_intents(query, expected_intent):
    plan = plan_query(query)
    assert plan.intent == expected_intent


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("  What   is   diabetes? ", "What is diabetes?"),
        ("what's diabetes?", "what is diabetes?"),
        ("Diabetes vs hypertension", "Diabetes versus hypertension"),
        ("A — B", "A - B"),
        ("A – B", "A - B"),
    ],
)
def test_query_normalization(raw, normalized):
    assert normalize_query(raw) == normalized


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


@pytest.mark.parametrize("query", [
    "What are the main findings?",
    "What is diabetes?",
    "Define hypertension.",
    "Résumez le document.",
    "ما هي النتائج الرئيسية؟",
])
def test_simple_queries_do_not_become_multihop(query):
    plan = plan_query(query)
    assert plan.needs_multi_hop is False


@pytest.mark.parametrize("message", [
    "I could not verify a sufficiently grounded answer from the indexed evidence; unsupported details were withheld.",
    "The indexed evidence was insufficient to safely perform the required clinical synthesis.",
    "The evidence was retrieved, but the required synthesis could not be verified without adding unsupported clinical content.",
    "I could not verify a sufficiently grounded answer from the indexed evidence.",
])
def test_control_messages_are_not_claims(message):
    assert is_control_message(message)
    verified = safe_verify_final_answer(message, [hit("Unrelated evidence.")])
    assert verified["checked"] is False
    assert verified["allow"] is False
    assert verified["claim_count"] == 0


@pytest.mark.parametrize(
    "answer,evidence",
    [
        ("Hyperglycemia was observed. [S1]", "Hyperglycemia was observed."),
        ("Albuminuria decreased. [S1]", "Albuminuria decreased."),
        ("The document reports diabetes. [S1]", "The document reports diabetes."),
        ("A normal value was reported. [S1]", "A normal value was reported."),
    ],
)
def test_supported_answers_are_still_verified(answer, evidence):
    result = safe_verify_final_answer(answer, [hit(evidence)])
    assert result["checked"] is True
    assert result["claim_count"] >= 1


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_answer_is_not_allowed(text):
    result = safe_verify_final_answer(text or "", [], require_entailment=True)
    assert result["allow"] is False
    assert result["reason"] in {"abstention_not_claim", "no_verifiable_claims"}


@pytest.mark.parametrize("query,evidence,expected_missing", [
    ("What is diabetes mellitus?", "Diabetes mellitus is a metabolic disorder.", False),
    ("What is dapagliflozin?", "Dapagliflozin is discussed.", False),
    ("What is albuminuria?", "The text discusses albuminuria.", False),
    ("What is diabetes?", "The text discusses hypertension.", True),
])
def test_entity_coverage_matches_actual_evidence(query, evidence, expected_missing):
    report = safe_score_entity_coverage(query, [hit(evidence)])
    assert bool(report["missing"]) is expected_missing


@pytest.mark.parametrize("query", [
    "What are the main findings?",
    "Summarize the document.",
    "Résumez les résultats.",
    "ما هي أهم النتائج؟",
])
def test_entity_coverage_entity_free_query_has_no_missing_penalty(query):
    report = safe_score_entity_coverage(query, [hit("Les résultats montrent une amélioration.")])
    assert report["entity_count"] == 0
    assert report["missing"] == []


def test_planner_labels_are_never_query_entities():
    report = safe_score_entity_coverage(
        "What are the main findings?",
        [hit("Les résultats montrent une amélioration.")],
        planned_entities=["relevant entities", "main findings", "planner"],
    )
    assert all(x not in {"relevant entities", "main findings", "planner"} for x in report["query_entities"])


@pytest.mark.parametrize(
    "value",
    [
        "user@example.com",
        "Call +213 555 123 456",
        "Patient ID 123456789",
        "record 987654321",
        "normal clinical text",
    ],
)
def test_sensitive_trace_redaction_is_deterministic(value):
    first = redact_sensitive_text(value)
    second = redact_sensitive_text(value)
    assert first == second


def test_sensitive_trace_redacts_email():
    assert "user@example.com" not in redact_sensitive_text("contact user@example.com")
    assert "[REDACTED_EMAIL]" in redact_sensitive_text("contact user@example.com")


def test_sensitive_trace_redacts_explicit_phone():
    result = redact_sensitive_text("Call +213 555 123 456")
    assert "555 123 456" not in result
    assert "[REDACTED_PHONE]" in result


def test_sensitive_trace_redacts_long_id():
    result = redact_sensitive_text("patient 1234567890")
    assert "1234567890" not in result
    assert "[REDACTED_ID]" in result


def test_sanitize_trace_does_not_mutate_input():
    source = {"question": "contact user@example.com", "score": 0.8}
    result = sanitize_trace(source)
    assert source["question"] == "contact user@example.com"
    assert "[REDACTED_EMAIL]" in result["question"]
    assert result["score"] == 0.8


@pytest.mark.parametrize("profile,ready", [
    ({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": True}, True),
    ({"feature_contract": False, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": True}, False),
    ({"feature_contract": True, "tests_green": False, "index_ready": True, "privacy_controls": True, "medical_safety": True}, False),
    ({"feature_contract": True, "tests_green": True, "index_ready": False, "privacy_controls": True, "medical_safety": True}, False),
    ({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": False, "medical_safety": True}, False),
    ({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": False}, False),
])
def test_production_readiness_all_gates(profile, ready):
    result = production_readiness(profile)
    assert result["release_ready"] is ready
    assert result["clinical_validation"] is False
    assert result["regulatory_approval"] is False


def test_feature_contract_is_complete_and_resolvable():
    result = validate_feature_contract()
    assert result["feature_count"] == 44
    assert result["unique_names"] is True
    assert result["all_resolved"] is True, result
    assert len(FEATURES) == 44


def test_runtime_contract_is_single_source_of_truth():
    contract = runtime_contract()
    assert contract["composition_root"] == "rag_project.application.create_rag_system"
    assert contract["canonical_service"] == CANONICAL_SERVICE
    assert contract["answer_pipeline_authority"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_pipeline_execution"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_monkey_patch"] is False
    assert contract["canonical_ingestion"].endswith("robust_ingest_file")
    assert contract["structured_request_context"] is True
    assert contract["structured_evidence_bundle"] is True
    assert contract["structured_answer_envelope"] is True
    assert contract["confidence_breakdown"] is True
    assert contract["ingestion_traceability"] is True
    assert contract["atomic_ingestion_publication"] is True
    assert contract["post_write_index_validation"] is True
    assert contract["production_contract_version"] == CONTRACT_VERSION
    assert contract["ingestion_contract_version"] == INGESTION_CONTRACT_VERSION


def test_canonical_runtime_installs_live_enhancer():
    result = install_canonical_runtime()
    assert result["canonical_service"] == CANONICAL_SERVICE
    assert result["answer_pipeline_authority"] == ANSWER_AUTHORITY
    assert result["class_binding_installed"] is True
    assert result["runtime_contract_bound"] is True


def test_pipeline_integrity_install_is_idempotent():
    install_pipeline_integrity()
    first = top_level_pipeline.rewrite_follow_up
    install_pipeline_integrity()
    second = top_level_pipeline.rewrite_follow_up
    assert first is second
    assert entity_coverage.extract_query_entities is not None
    assert final_answer_contract.verify_final_answer is not None
    assert god_mode_100.verify_final_answer is not None


def test_ingestion_request_id_format():
    request_id = new_ingestion_request_id()
    assert re.fullmatch(r"ing-[0-9a-f]{16}", request_id)


def test_ingestion_request_ids_are_unique():
    values = {new_ingestion_request_id() for _ in range(100)}
    assert len(values) == 100


def test_ingestion_wrapper_preserves_result_and_adds_trace():
    def original(system, pdf_path, *args, **kwargs):
        assert pdf_path == Path("doc.pdf")
        return {"status": "READY", "chunks": 12}

    wrapped = wrap_ingest_callable(original)
    result = wrapped(object(), Path("doc.pdf"))
    assert result["status"] == "READY"
    assert result["chunks"] == 12
    assert re.fullmatch(r"ing-[0-9a-f]{16}", result["ingestion_request_id"])
    assert result["ingestion_trace_version"] == INGESTION_CONTRACT_VERSION
    assert result["ingestion_elapsed_ms"] >= 0
    assert result["ingestion_contract"]["atomic_publication"] is True
    assert result["ingestion_contract"]["lease_fencing"] is True
    assert result["ingestion_contract"]["post_write_validation"] is True


def test_ingestion_wrapper_preserves_non_dict_result():
    def original(system, pdf_path):
        return True

    wrapped = wrap_ingest_callable(original)
    assert wrapped(object(), "x.pdf") is True


def test_ingestion_wrapper_is_idempotent():
    def original(system, pdf_path):
        return {"ok": True}

    first = wrap_ingest_callable(original)
    second = wrap_ingest_callable(first)
    assert second is first


def test_ingestion_wrapper_reraises_errors():
    def original(system, pdf_path):
        raise RuntimeError("boom")

    wrapped = wrap_ingest_callable(original)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(SimpleNamespace(logger=None), "x.pdf")


@pytest.mark.parametrize("score", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_confidence_calibration_is_bounded(score):
    result = calibrate_confidence(
        retrieval=score,
        rerank=score,
        entailment=score,
        entity_coverage=score,
        source_agreement=score,
        contradiction=0.0,
        safety_conflict=0.0,
    )
    assert 0.0 <= result.calibrated <= 1.0
    assert result.level in {"LOW", "MEDIUM", "HIGH"}


def test_confidence_calibration_drops_on_contradiction():
    clean = calibrate_confidence(0.9, 0.9, 0.9, 1.0, 0.9, 0.0, 0.0)
    conflicted = calibrate_confidence(0.9, 0.9, 0.9, 1.0, 0.9, 1.0, 0.0)
    assert conflicted.calibrated < clean.calibrated


def test_confidence_calibration_drops_on_safety_conflict():
    clean = calibrate_confidence(0.9, 0.9, 0.9, 1.0, 0.9, 0.0, 0.0)
    risky = calibrate_confidence(0.9, 0.9, 0.9, 1.0, 0.9, 0.0, 1.0)
    assert risky.calibrated < clean.calibrated


def test_top_level_simple_query_has_clean_contract():
    system = FakeSystem(hits=[hit("The main findings include hyperglycemia and ketoacidosis.")], llm=None)
    result = top_level_pipeline.complete_phases(
        system,
        "What are the main findings?",
        {"hits": system.retriever.hits, "answer": "The main findings include hyperglycemia and ketoacidosis. [S1]"},
    )
    assert result["rewritten_question"] == "What are the main findings?"
    assert result["phase_plan"]["entities"] == ()
    assert result["phase_plan"]["needs_multi_hop"] is False
    assert result["extractive_stage"]["supported"] is True
    assert result["answer"]
    assert result.get("status") != "GENERATION_ABSTAIN"


def test_top_level_retrieval_records_calls_for_hard_query():
    system = FakeSystem(hits=[hit("Metformin dose is discussed in the document.")], llm=None)
    result = top_level_pipeline.complete_phases(
        system,
        "What is the dose of metformin?",
        {"hits": []},
    )
    assert result["adaptive_retrieval"]["escalated"] is True
    assert system.retriever.calls
    assert any("metformin" in call["query"].casefold() for call in system.retriever.calls)


def test_top_level_extractive_path_preserves_source_markers():
    system = FakeSystem(hits=[hit("Hyperglycemia was observed.")], llm=None)
    result = top_level_pipeline.complete_phases(system, "What was observed?", {"hits": system.retriever.hits})
    assert "[S1]" in result["answer"] or "[S1]" in result["extractive_stage"]["draft"]


@pytest.mark.parametrize(
    "query",
    [
        "What is 500 mg?",
        "What is the dose of metformin?",
        "What is the dosage range?",
        "What percentage was reported?",
        "What is the value in mmol/L?",
    ],
)
def test_numeric_queries_request_numeric_precision(query):
    plan = plan_query(query)
    assert plan.needs_numeric is True


@pytest.mark.parametrize(
    "query",
    [
        "What is in the table?",
        "Which row contains the value?",
        "Show the table entries.",
        "Quels sont les résultats du tableau ?",
    ],
)
def test_table_queries_are_detected(query):
    plan = plan_query(query)
    assert plan.needs_table is True


@pytest.mark.parametrize(
    "query",
    [
        "What does Figure 1 show?",
        "Which diagram is referenced?",
        "Show the chart.",
        "Que montre la figure ?",
    ],
)
def test_figure_queries_are_detected(query):
    plan = plan_query(query)
    assert plan.needs_figure is True


@pytest.mark.parametrize("query", [
    "What are the contraindications of metformin?",
    "How should DKA be managed?",
    "What causes cirrhosis?",
    "What is the mechanism of insulin?",
    "What is the prognosis?",
])
def test_high_risk_medical_queries_trigger_strict_mode(query):
    plan = plan_query(query)
    assert plan.intent in {"management", "etiology", "mechanism", "prognosis"} or plan.needs_multi_hop


def test_evidence_provenance_survives_entity_matching():
    first = hit("Diabetes is a metabolic condition.", page=3)
    second = hit("Hypertension is common.", page=9)
    report = safe_score_entity_coverage("What is diabetes?", [first, second])
    assert report["matches"]
    assert "S1" in report["matches"][0]["evidence_sources"]


def test_no_synthetic_contradiction_on_abstention():
    result = safe_verify_final_answer(
        "The indexed evidence was insufficient to safely perform the required clinical synthesis.",
        [hit("Hypertension is common.")],
    )
    assert result["checked"] is False
    assert result["blocked_claims"] == 0
    assert result["matrix_claim_count"] == 0


def test_feature_targets_are_unique():
    names = [feature.name for feature in FEATURES]
    assert len(names) == len(set(names))
    targets = [feature.target for feature in FEATURES]
    assert all(":" in target for target in targets)


@pytest.mark.parametrize("language_query", [
    "What is diabetes?",
    "Qu'est-ce que le diabète ?",
    "ما هو مرض السكري؟",
])
def test_multilingual_queries_remain_nonempty_after_normalization(language_query):
    result = normalize_query(language_query)
    assert result


@pytest.mark.parametrize("query", [
    "diabetes complications",
    "hypertension treatment",
    "dka diagnosis",
    "albuminuria outcome",
    "metformin dose",
])
def test_query_decomposition_never_returns_empty_strings(query):
    pieces = decompose_query(query)
    assert pieces
    assert all(piece.strip() for piece in pieces)


def test_runtime_contract_versions_are_consistent():
    contract = runtime_contract()
    assert contract["production_contract_version"] == CONTRACT_VERSION
    assert contract["ingestion_contract_version"] == INGESTION_CONTRACT_VERSION


def test_canonical_authority_constants_match():
    assert ANSWER_AUTHORITY == ANSWER_PIPELINE_AUTHORITY
    assert CANONICAL_SERVICE == "rag_project.app.production_rag.ProductionRAGSystem"


def test_safe_entity_extraction_filters_protocol_terms():
    query = "What are the main findings? Relevant entities: diabetes mellitus"
    result = safe_extract_query_entities(query, planned_entities=["relevant entities", "main findings"])
    assert "relevant entities" not in result
    assert "main findings" not in result
    assert "diabetes mellitus" in result


def test_safe_followup_does_not_duplicate_context_on_repeated_application():
    history = [("What are the complications of diabetes?", "Diabetic nephropathy is discussed.")]
    once = safe_rewrite_follow_up("What about this?", history)
    twice = safe_rewrite_follow_up(once, history)
    assert "Follow-up:" not in twice
    assert "Relevant entities:" not in twice


def test_runtime_contract_has_medical_safety_gate():
    assert runtime_contract()["medical_safety_gate"] is True


def test_runtime_contract_has_request_traceability():
    contract = runtime_contract()
    assert contract["request_traceability"] is True
    assert contract["ingestion_traceability"] is True


@pytest.mark.parametrize("query", [
    "What is diabetes mellitus?",
    "What is dapagliflozin?",
    "What is albuminuria?",
    "What is CKD?",
    "What is DKA?",
])
def test_medical_queries_have_at_least_one_signal(query):
    plan = plan_query(query)
    assert plan.entities or plan.intent in {"factual", "definition"}


@pytest.mark.parametrize("query", [
    "What are the main findings?",
    "What is diabetes?",
    "What does the article report?",
])
def test_simple_query_variant_list_is_bounded(query):
    plan = plan_query(query)
    assert 1 <= len(plan.variants) <= 10


def test_production_contract_does_not_claim_clinical_validation():
    result = production_readiness({
        "feature_contract": True,
        "tests_green": True,
        "index_ready": True,
        "privacy_controls": True,
        "medical_safety": True,
    })
    assert result["clinical_validation"] is False
    assert result["regulatory_approval"] is False


def test_control_message_detection_is_case_insensitive():
    assert is_control_message("I COULD NOT VERIFY A SUFFICIENTLY GROUNDED ANSWER FROM THE INDEXED EVIDENCE.")


def test_normalization_handles_excess_whitespace_without_losing_text():
    result = normalize_query("What   is   the   mechanism   of   insulin?")
    assert "mechanism" in result.casefold()
    assert "insulin" in result.casefold()


def test_numeric_plan_has_low_temperature_compatibility():
    plan = top_level_pipeline.deterministic_phase1("What is the dose of metformin?")
    temperature = top_level_pipeline.dynamic_temperature(plan, default=0.2)
    assert temperature == 0.0


def test_simple_factual_plan_allows_normal_temperature():
    plan = top_level_pipeline.deterministic_phase1("What are the main findings?")
    temperature = top_level_pipeline.dynamic_temperature(plan, default=0.2)
    assert 0.0 <= temperature <= 0.2


def test_empty_retrieval_does_not_crash_entity_coverage():
    result = safe_score_entity_coverage("What is diabetes?", [])
    assert result["entity_count"] >= 1
    assert result["coverage"] == 0.0


def test_empty_retrieval_produces_no_fake_matches():
    result = safe_score_entity_coverage("What is diabetes?", [])
    assert result["matches"] == []


def test_planner_explicit_nonmedical_labels_do_not_pollute_safe_entity_layer():
    entities = safe_extract_query_entities(
        "What are the main findings?",
        planned_entities=("intent", "planner confidence", "conversation context"),
    )
    assert entities == ()


def test_source_marker_detection_does_not_classify_source_answer_as_control():
    assert is_control_message("Hyperglycemia was observed. [S1]") is False


def test_feature_contract_has_44_entries_for_release_certification():
    assert len(FEATURES) == 44


def test_pipeline_authority_is_top_level_complete_phases():
    assert ANSWER_PIPELINE_AUTHORITY == "rag_project.intelligence.top_level_pipeline.complete_phases"


def test_ingestion_contract_contains_atomic_publication_guarantee():
    def original(system, pdf_path):
        return {"status": "READY"}

    result = wrap_ingest_callable(original)(object(), "document.pdf")
    contract = result["ingestion_contract"]
    assert contract["atomic_publication"] is True
    assert contract["post_write_validation"] is True


def test_ingestion_trace_has_elapsed_time():
    def original(system, pdf_path):
        return {"status": "READY"}

    result = wrap_ingest_callable(original)(object(), "document.pdf")
    assert isinstance(result["ingestion_elapsed_ms"], (int, float))
    assert result["ingestion_elapsed_ms"] >= 0


def test_entity_coverage_returns_structured_breakdown():
    result = safe_score_entity_coverage("What is diabetes?", [hit("Diabetes is a metabolic disorder.")])
    for key in ("query_entities", "entity_count", "covered", "missing", "partial", "coverage", "partial_coverage", "per_entity", "matches"):
        assert key in result


def test_final_verification_returns_structured_breakdown():
    result = safe_verify_final_answer("Hyperglycemia was observed. [S1]", [hit("Hyperglycemia was observed.")])
    for key in ("checked", "allow", "reason", "claim_count", "blocked_claims", "supported_ratio", "matrix_claim_count", "matrix_all_entailed"):
        assert key in result


@pytest.mark.parametrize("value", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_confidence_level_is_always_valid(value):
    result = calibrate_confidence(value, value, value, value, value, 0.0, 0.0)
    assert result.level in {"LOW", "MEDIUM", "HIGH"}


def test_hard_query_retrieval_escalates_and_limits_calls():
    system = FakeSystem(hits=[hit("Dose information 500 mg.")])
    result = top_level_pipeline.complete_phases(system, "What is the dose of metformin?", {"hits": []})
    assert result["adaptive_retrieval"]["escalated"] is True
    assert len(system.retriever.calls) <= 8


def test_simple_query_retrieval_can_use_initial_hits():
    system = FakeSystem(hits=[])
    initial = [hit("The document reports hyperglycemia.")]
    result = top_level_pipeline.complete_phases(system, "What was reported?", {"hits": initial})
    assert result["hits"] == initial or result["extractive_stage"]["supported"] is True


def test_pipeline_output_contains_phase_visibility():
    system = FakeSystem(hits=[hit("Hyperglycemia was reported.")])
    result = top_level_pipeline.complete_phases(system, "What was reported?", {"hits": system.retriever.hits})
    assert "phases" in result
    for key in ("phase_1_query_understanding", "phase_2_retrieval_precision", "phase_3_two_stage_generation", "phase_4_verification", "phase_5_intelligence_visibility"):
        assert key in result["phases"]


def test_pipeline_output_contains_retrieval_diagnostics():
    system = FakeSystem(hits=[hit("Hyperglycemia was reported.")])
    result = top_level_pipeline.complete_phases(system, "What was reported?", {"hits": system.retriever.hits})
    diagnostics = result["adaptive_retrieval"]
    assert "stage" in diagnostics
    assert "queries" in diagnostics
    assert "final_hits" in diagnostics


def test_pipeline_output_contains_compression_diagnostics():
    system = FakeSystem(hits=[hit("Hyperglycemia was reported. It was associated with ketoacidosis.")])
    result = top_level_pipeline.complete_phases(system, "What was reported?", {"hits": system.retriever.hits})
    assert "context_compression" in result
    assert "compression_ratio" in result["context_compression"]


def test_pipeline_output_contains_generation_policy():
    system = FakeSystem(hits=[hit("Hyperglycemia was reported.")])
    result = top_level_pipeline.complete_phases(system, "What was reported?", {"hits": system.retriever.hits})
    assert "two_stage_policy" in result
    assert "required" in result["two_stage_policy"]


def test_runtime_binding_can_be_called_repeatedly():
    first = install_canonical_runtime()
    second = install_canonical_runtime()
    assert first["class_binding_installed"] is True
    assert second["class_binding_installed"] is True


@pytest.mark.parametrize("query", [
    "What is the relationship between diabetes and albuminuria?",
    "Compare diabetes and hypertension.",
    "What causes diabetic nephropathy?",
])
def test_relationship_queries_are_marked_complex(query):
    plan = plan_query(query)
    assert plan.needs_multi_hop is True


def test_control_message_does_not_require_evidence():
    result = safe_verify_final_answer("The indexed evidence was insufficient to safely perform the required clinical synthesis.", [])
    assert result["checked"] is False
    assert result["matrix_claim_count"] == 0


def test_evidence_free_answer_fails_closed():
    result = safe_verify_final_answer("Diabetes is present. [S1]", [])
    assert result["allow"] is False


def test_unrelated_evidence_cannot_create_supported_claim():
    result = safe_verify_final_answer("Diabetes is present. [S1]", [hit("Hypertension is common.")])
    assert result["allow"] is False


def test_query_entities_do_not_include_followup_protocol_text():
    entities = safe_extract_query_entities("diabetes Follow-up: Relevant entities:")
    assert all("follow-up" not in e for e in entities)
    assert all("relevant entities" not in e for e in entities)


def test_safe_rewrite_preserves_user_question_text():
    question = "What are the complications of diabetes?"
    result = safe_rewrite_follow_up(question, [])
    assert result == question


def test_safe_rewrite_handles_missing_history():
    assert safe_rewrite_follow_up("What about this?", None) == "What about this?"


def test_safe_rewrite_handles_empty_question():
    assert safe_rewrite_follow_up("", [("What is diabetes?", "Diabetes is a condition.")]) == ""


def test_feature_contract_resolution_has_no_duplicates():
    result = validate_feature_contract()
    assert result["duplicates"] == []


def test_feature_contract_unresolved_is_empty():
    result = validate_feature_contract()
    assert result["unresolved"] == {}


@pytest.mark.parametrize("query", [
    "What is the dose and contraindications of metformin?",
    "Compare diabetes and hypertension.",
    "What causes and how is cirrhosis managed?",
])
def test_complex_plan_has_bounded_variants(query):
    plan = plan_query(query)
    assert 1 <= len(plan.variants) <= 10


@pytest.mark.parametrize("query", [
    "What is diabetes?",
    "Define diabetes.",
    "What does the document say about diabetes?",
])
def test_factual_queries_have_safe_defaults(query):
    plan = plan_query(query)
    assert plan.needs_numeric is False
    assert plan.needs_table is False
    assert plan.needs_figure is False


def test_hard_query_plan_is_explicit_about_numeric_need():
    plan = plan_query("What is the dose of metformin?")
    assert plan.needs_numeric is True


def test_pipeline_integrity_install_does_not_change_class_authority():
    install_pipeline_integrity()
    assert ANSWER_PIPELINE_AUTHORITY == "rag_project.intelligence.top_level_pipeline.complete_phases"


def test_contract_contains_single_canonical_ingestion():
    contract = runtime_contract()
    assert contract["canonical_ingestion"] == "rag_project.ingestion.robust_ingestor.robust_ingest_file"


def test_privacy_sanitizer_preserves_non_sensitive_metadata():
    data = {"question": "What is diabetes?", "page": 12, "score": 0.81}
    result = sanitize_trace(data)
    assert result["page"] == 12
    assert result["score"] == 0.81
    assert result["question"] == data["question"]


def test_production_readiness_requires_every_gate():
    complete = {key: True for key in ("feature_contract", "tests_green", "index_ready", "privacy_controls", "medical_safety")}
    for missing in complete:
        profile = dict(complete)
        profile[missing] = False
        assert production_readiness(profile)["release_ready"] is False


@pytest.mark.parametrize("score", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_confidence_never_exits_unit_interval(score):
    result = calibrate_confidence(score, score, score, score, score, 0.0, 0.0)
    assert 0.0 <= result.calibrated <= 1.0


def test_abstention_reason_is_structured():
    result = safe_verify_final_answer("The evidence was retrieved, but the required synthesis could not be verified without adding unsupported clinical content.", [hit("Evidence")])
    assert result["reason"] == "abstention_not_claim"


def test_pipeline_direct_summary_does_not_inject_internal_labels():
    system = FakeSystem(hits=[hit("The principal finding was hyperglycemia.")])
    result = top_level_pipeline.complete_phases(system, "What are the main findings?", {"hits": system.retriever.hits})
    query = result["rewritten_question"]
    assert "Follow-up:" not in query
    assert "Relevant entities:" not in query


def test_pipeline_direct_summary_reports_zero_entities():
    system = FakeSystem(hits=[hit("The principal finding was hyperglycemia.")])
    result = top_level_pipeline.complete_phases(system, "What are the main findings?", {"hits": system.retriever.hits})
    assert result["phase_plan"]["entities"] == ()


def test_pipeline_direct_summary_reports_no_multihop():
    system = FakeSystem(hits=[hit("The principal finding was hyperglycemia.")])
    result = top_level_pipeline.complete_phases(system, "What are the main findings?", {"hits": system.retriever.hits})
    assert result["phase_plan"]["needs_multi_hop"] is False


def test_runtime_contract_is_json_serializable():
    import json

    json.dumps(runtime_contract())


def test_production_readiness_is_json_serializable():
    import json

    result = production_readiness({"feature_contract": True, "tests_green": True, "index_ready": True, "privacy_controls": True, "medical_safety": True})
    json.dumps(result)


def test_ingestion_result_is_json_serializable():
    import json

    def original(system, pdf_path):
        return {"status": "READY", "count": 1}

    result = wrap_ingest_callable(original)(object(), "doc.pdf")
    json.dumps(result)


def test_entity_report_is_json_serializable():
    import json

    result = safe_score_entity_coverage("What is diabetes?", [hit("Diabetes is a metabolic disorder.")])
    json.dumps(result, default=str)


def test_final_verification_is_json_serializable():
    import json

    result = safe_verify_final_answer("Hyperglycemia was observed. [S1]", [hit("Hyperglycemia was observed.")])
    json.dumps(result, default=str)


@pytest.mark.parametrize("query", [
    "What is diabetes?",
    "What is dapagliflozin?",
    "What is albuminuria?",
])
def test_query_plan_original_is_preserved(query):
    plan = plan_query(query)
    assert plan.original == query


@pytest.mark.parametrize("query", [
    "What is diabetes?",
    "Define hypertension.",
    "What was observed?",
])
def test_simple_query_plan_has_single_subquery(query):
    plan = plan_query(query)
    assert len(plan.subqueries) == 1


@pytest.mark.parametrize("query", [
    "What is the dose of metformin?",
    "What are the contraindications of metformin?",
])
def test_medical_hard_queries_keep_temperature_zero_in_top_level_policy(query):
    plan = top_level_pipeline.deterministic_phase1(query)
    assert top_level_pipeline.dynamic_temperature(plan, 0.2) == 0.0


@pytest.mark.parametrize("query", [
    "What is diabetes?",
    "What are the main findings?",
])
def test_simple_factual_query_never_contains_protocol_labels_after_full_rewrite(query):
    result = safe_rewrite_follow_up(query, [])
    assert all(label not in result.casefold() for label in ("follow-up:", "relevant entities:", "query entities:"))


def test_feature_contract_release_rule_requires_44_features():
    result = validate_feature_contract()
    assert result["feature_count"] == 44
    assert result["all_resolved"]
