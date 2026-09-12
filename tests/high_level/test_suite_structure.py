from pathlib import Path

import pytest


EXPECTED = {
    "01_ingestion": {
        "test_basic_ingestion.py",
        "test_duplicate_and_atomic.py",
        "test_large_document_resilience.py",
        "test_ocr_and_scanned.py",
        "test_page_and_metadata_integrity.py",
        "test_partial_failure_atomicity.py",
        "test_publication_boundary.py",
        "test_versioned_replacement.py",
    },
    "02_retrieval": {
        "test_document_aware_filtering.py",
        "test_hybrid_and_exact_terms.py",
        "test_multi_query_and_cascade.py",
        "test_ranking_quality.py",
        "test_table_figure_numeric.py",
    },
    "03_query_intelligence": {
        "test_decomposition.py",
        "test_entity_extraction.py",
        "test_followup_isolation.py",
        "test_intent_and_complexity.py",
    },
    "04_answer_paths": {
        "test_citation_and_numeric_integrity.py",
        "test_constrained_llm_path.py",
        "test_extractive_fast_path.py",
        "test_recovery_and_abstention.py",
        "test_template_slot_filling.py",
    },
    "05_grounding_safety": {
        "test_claim_verification.py",
        "test_contradiction_and_safety.py",
        "test_grounding_envelope.py",
        "test_prompt_injection_resistance.py",
    },
    "06_latency_performance": {
        "test_answer_budget_contract.py",
        "test_complex_query_ceiling.py",
        "test_no_unnecessary_llm_calls.py",
        "test_simple_query_latency.py",
    },
    "07_multilingual": {
        "test_answer_language.py",
        "test_cross_language_retrieval.py",
        "test_en_fr_ar_routing.py",
    },
    "08_storage_index": {
        "test_cache_freshness.py",
        "test_cleanup_and_isolation.py",
        "test_index_identity_and_ready.py",
        "test_ready_boundary.py",
        "test_ready_only_retrieval_boundary.py",
        "test_search_visibility.py",
    },
    "09_conversation": {
        "test_conversation_memory.py",
        "test_session_isolation.py",
    },
    "10_security_privacy": {
        "test_data_redaction.py",
        "test_input_boundary.py",
        "test_network_and_payload_limits.py",
        "test_runtime_trace_redaction.py",
        "test_security_boundaries.py",
    },
    "11_resilience": {
        "test_embedding_timeout_contract.py",
        "test_failure_modes.py",
        "test_retrieval_failure.py",
        "test_safe_degrade_contract.py",
        "test_verified_generation_recovery.py",
    },
    "12_production_contracts": {
        "test_production_contracts.py",
        "test_runtime_success_contract.py",
    },
    "13_end_to_end": {
        "test_scenario_adversarial.py",
        "test_scenario_failure_recovery.py",
        "test_scenario_followup_conversation.py",
        "test_scenario_insufficient_evidence.py",
        "test_scenario_large_document.py",
        "test_scenario_multi_document_library.py",
        "test_scenario_multilingual.py",
        "test_scenario_numeric_table.py",
        "test_scenario_scanned_mixed.py",
        "test_scenario_simple_textbook.py",
    },
}

ROOT_TESTS = {
    "test_gold_regression.py",
    "test_pdf_system_behavior.py",
    "test_suite_structure.py",
}


@pytest.mark.high_level
def test_high_level_suite__contains_exactly_the_planned_behavior_files():
    root = Path(__file__).resolve().parent

    actual_directories = {path.name for path in root.iterdir() if path.is_dir() and path.name != "__pycache__"}
    assert actual_directories == set(EXPECTED), (actual_directories, set(EXPECTED))

    for directory, expected_files in EXPECTED.items():
        actual = root / directory
        actual_files = {path.name for path in actual.glob("test_*.py")}
        assert actual_files == expected_files, f"{directory}: expected {sorted(expected_files)}, got {sorted(actual_files)}"

    actual_root_tests = {path.name for path in root.glob("test_*.py")}
    assert actual_root_tests == ROOT_TESTS, (actual_root_tests, ROOT_TESTS)
