from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    [
        "rag_project.application",
        "rag_project.runtime",
        "rag_project.runtime_final_gate",
        "rag_project.runtime_quality_gate",
        "rag_project.runtime_recovery",
        "rag_project.runtime_hardening",
        "rag_project.security",
        "rag_project.configuration.settings",
        "rag_project.chunking.semantic_chunker",
        "rag_project.parsing.pdf_extractor",
        "rag_project.ocr.ocr_service",
        "rag_project.ingestion.atomic_claim",
        "rag_project.ingestion.robust_ingestor",
        "rag_project.ingestion.responsive_supervisor",
        "rag_project.ingestion.state_store",
        "rag_project.embeddings.embedding_service",
        "rag_project.embeddings.embedding_runtime",
        "rag_project.storage.vector_store",
        "rag_project.storage.vector_store_runtime",
        "rag_project.retrieval.metadata_filter",
        "rag_project.retrieval.query_rewriter",
        "rag_project.retrieval.hybrid_retriever",
        "rag_project.reranking.reranker",
        "rag_project.intelligence.query_intelligence",
        "rag_project.intelligence.semantic_reasoning",
        "rag_project.intelligence.adaptive_retrieval",
        "rag_project.intelligence.entity_coverage",
        "rag_project.intelligence.evidence_guard",
        "rag_project.intelligence.evidence_entailment",
        "rag_project.intelligence.confidence_calibration",
        "rag_project.intelligence.final_answer_contract",
        "rag_project.intelligence.top_level_pipeline",
        "rag_project.intelligence.god_mode_100",
        "rag_project.intelligence.production_contract",
        "rag_project.intelligence.medical_safety",
        "rag_project.intelligence.ui_visibility_contract",
    ],
)
def test_critical_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_application_contract_is_single_authority() -> None:
    from rag_project.application import ANSWER_PIPELINE_AUTHORITY, runtime_contract

    contract = runtime_contract()
    assert ANSWER_PIPELINE_AUTHORITY.endswith("top_level_pipeline.complete_phases")
    assert contract["answer_pipeline_authority"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_pipeline_execution"] == ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_monkey_patch"] is False
    assert contract["quality_policy"]
    assert contract["security_policy"]
    assert contract["storage_policy"]
    assert contract["runtime_policy"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("1", True),
        ("on", True),
        ("false", False),
        ("no", False),
        ("0", False),
        (None, False),
        ("random", False),
    ],
)
def test_settings_boolean_parser_boundaries(raw, expected) -> None:
    from rag_project.configuration.settings import Settings

    assert Settings._parse_bool(raw, False) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("3.5", 3.5), (3, 3.0), ("bad", 7.0), (None, 7.0), ("", 7.0)],
)
def test_settings_float_parser_boundaries(raw, expected) -> None:
    from rag_project.configuration.settings import Settings

    assert Settings._parse_float(raw, 7.0) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("3", 3), (3, 3), ("bad", 7), (None, 7), ("", 7)],
)
def test_settings_int_parser_boundaries(raw, expected) -> None:
    from rag_project.configuration.settings import Settings

    assert Settings._parse_int(raw, 7) == expected


def test_settings_normalize_core_ranges() -> None:
    from rag_project.configuration.settings import Settings

    settings = Settings(
        chunk_size=10,
        chunk_overlap=999,
        top_k=999,
        temperature=99,
        vector_weight=-3,
        embedding_batch_size=999,
        embedding_retries=99,
        embedding_timeout_seconds=1,
        generation_max_output_tokens=1,
        context_token_budget=1,
        max_workers=0,
        max_memory_target=1,
        ollama_concurrency=0,
        ingestion_lease_seconds=1,
        retrieval_candidate_multiplier=99,
        evidence_min_confidence=-1,
        max_query_variants=99,
    )
    assert settings.chunk_size >= 200
    assert 0 <= settings.chunk_overlap < settings.chunk_size
    assert 1 <= settings.top_k <= 50
    assert 0 <= settings.temperature <= 1
    assert 0 <= settings.vector_weight <= 1
    assert 1 <= settings.embedding_batch_size <= 32
    assert 0 <= settings.embedding_retries <= 3
    assert 30 <= settings.embedding_timeout_seconds <= 300
    assert settings.generation_max_output_tokens >= 32
    assert settings.context_token_budget >= 256
    assert settings.max_workers >= 1
    assert settings.max_memory_target >= 256
    assert settings.ollama_concurrency >= 1
    assert settings.ingestion_lease_seconds >= 30
    assert 2 <= settings.retrieval_candidate_multiplier <= 12
    assert 0 <= settings.evidence_min_confidence <= 1
    assert 1 <= settings.max_query_variants <= 12


def test_text_pipeline_normalizes_unicode_whitespace_and_arabic() -> None:
    from rag_project.utils.text_utils import clean_text, normalize_arabic, normalize_whitespace

    assert normalize_whitespace("  alpha\n\n beta  ") == "alpha beta"
    assert clean_text("a\u00a0b\r\n\r\n\r\nc") == "a b\n\nc"
    assert normalize_arabic("إيمان آية ى") == "ايمان ايه ي"


@pytest.mark.parametrize(
    ("text", "language"),
    [("diabetes treatment", "en"), ("le traitement", "fr"), ("علاج السكري", "ar"), ("12345", "unknown"), ("", "unknown")],
)
def test_language_detection_contract(text: str, language: str) -> None:
    from rag_project.utils.text_utils import detect_language

    assert detect_language(text) == language


@pytest.mark.parametrize(
    "text",
    [
        "Page 5",
        "p. 12",
        "Page: 34 of 100",
        "7/100",
        "7 - of 100",
    ],
)
def test_page_number_extraction_accepts_supported_forms(text: str) -> None:
    from rag_project.utils.text_utils import extract_page_number

    assert extract_page_number(text) is not None


def test_page_number_extraction_rejects_garbage() -> None:
    from rag_project.utils.text_utils import extract_page_number

    assert extract_page_number("") is None
    assert extract_page_number("not a page header") is None


@pytest.mark.parametrize(
    ("query", "document", "minimum"),
    [
        ("diabetes", "diabetes nephropathy", 0.5),
        ("diabète", "diabetes", 0.1),
        ("kidney renal", "renal disease", 0.2),
        ("unseen concept", "completely unrelated text", 0.0),
        ("", "document", 0.0),
    ],
)
def test_keyword_overlap_is_stable(query: str, document: str, minimum: float) -> None:
    from rag_project.utils.text_utils import keyword_overlap_score

    score = keyword_overlap_score(query, document)
    assert 0 <= score <= 1
    assert score >= minimum


def test_metadata_filter_always_preserves_ready_state() -> None:
    from rag_project.retrieval.metadata_filter import MetadataFilter

    assert MetadataFilter.build(None) == {"index_state": "READY"}
    assert MetadataFilter.build({}) == {"index_state": "READY"}
    assert MetadataFilter.build({"faculty": ""}) == {"index_state": "READY"}
    assert MetadataFilter.build({"faculty": "med"}) == {"$and": [{"index_state": "READY"}, {"faculty": "med"}]}
    assert MetadataFilter.build({"index_state": "READY", "faculty": "med"}) == {"index_state": "READY", "faculty": "med"}


def test_conversation_memory_is_bounded() -> None:
    from rag_project.retrieval.query_rewriter import ConversationMemory

    memory = ConversationMemory(max_history=2)
    memory.add("q1", "a1")
    memory.add("q2", "a2")
    memory.add("q3", "a3")
    assert memory.history == [("q2", "a2"), ("q3", "a3")]
    assert "q1" not in memory.prompt_context()
    assert "q3" in memory.prompt_context()


def test_query_rewriter_empty_and_follow_up_fallbacks() -> None:
    from rag_project.retrieval.query_rewriter import QueryRewriter

    assert QueryRewriter.rewrite("   ") == ""
    rewritten = QueryRewriter.rewrite("what about this?", [("What is diabetes?", "Diabetes is a metabolic disease.")])
    assert "What is diabetes?" in rewritten
    assert "what about this?" in rewritten


@pytest.mark.parametrize(
    ("answer", "expected_count"),
    [("", 0), ("OK", 0), ("Yes. Diabetes is chronic.", 1), ("- one\n- two", 2)],
)
def test_claim_splitting_filters_tiny_noise(answer: str, expected_count: int) -> None:
    from rag_project.intelligence.evidence_guard import split_claims

    assert len(split_claims(answer)) == expected_count


def test_final_answer_verification_fails_closed_on_no_claims() -> None:
    from rag_project.intelligence.final_answer_contract import verify_final_answer

    result = verify_final_answer("", [], require_entailment=True)
    assert result["checked"] is False
    assert result["allow"] is False
    assert result["reason"] == "no_verifiable_claims"


def test_confidence_calibration_is_bounded_and_monotonic_for_good_inputs() -> None:
    from rag_project.intelligence.confidence_calibration import calibrate_confidence

    low = calibrate_confidence(
        retrieval=0,
        rerank=0,
        entailment=0,
        entity_coverage=0,
        source_agreement=0,
        contradiction=1,
        safety_conflict=1,
    )
    high = calibrate_confidence(
        retrieval=1,
        rerank=1,
        entailment=1,
        entity_coverage=1,
        source_agreement=1,
        contradiction=0,
        safety_conflict=0,
    )
    assert 0 <= low.calibrated <= 1
    assert 0 <= high.calibrated <= 1
    assert high.calibrated > low.calibrated
    assert high.level in {"medium", "high"}


@pytest.mark.parametrize("bad", [-1.0, 2.0])
def test_confidence_calibration_clamps_inputs(bad: float) -> None:
    from rag_project.intelligence.confidence_calibration import calibrate_confidence

    value = calibrate_confidence(
        retrieval=bad,
        rerank=bad,
        entailment=bad,
        entity_coverage=bad,
        source_agreement=bad,
        contradiction=bad,
        safety_conflict=bad,
    )
    assert all(0 <= item <= 1 for item in value.factors.values())
    assert 0 <= value.calibrated <= 1


@pytest.mark.parametrize(
    "path",
    [
        "app.py",
        "rag_project/app/bookrag_ui.py",
        "rag_project/app/intelligence_panel.py",
        "rag_project/app/production_rag.py",
        "rag_project/intelligence/god_mode_100.py",
        "rag_project/intelligence/top_level_pipeline.py",
        "rag_project/intelligence/final_answer_contract.py",
        "rag_project/security.py",
    ],
)
def test_critical_source_files_are_nonempty(path: str) -> None:
    target = ROOT / path
    assert target.is_file()
    assert target.stat().st_size > 100
