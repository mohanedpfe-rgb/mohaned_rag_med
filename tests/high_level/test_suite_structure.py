from pathlib import Path

import pytest


EXPECTED = {
    "01_ingestion": ["test_basic_ingestion.py", "test_duplicate_and_atomic.py", "test_ocr_and_scanned.py", "test_large_document_resilience.py", "test_page_and_metadata_integrity.py"],
    "02_retrieval": ["test_hybrid_and_exact_terms.py", "test_table_figure_numeric.py", "test_multi_query_and_cascade.py", "test_document_aware_filtering.py"],
    "03_query_intelligence": ["test_intent_and_complexity.py", "test_entity_extraction.py", "test_followup_isolation.py", "test_decomposition.py"],
    "04_answer_paths": ["test_extractive_fast_path.py", "test_template_slot_filling.py", "test_constrained_llm_path.py", "test_recovery_and_abstention.py", "test_citation_and_numeric_integrity.py"],
    "05_grounding_safety": ["test_claim_verification.py", "test_contradiction_and_safety.py", "test_prompt_injection_resistance.py"],
    "06_latency_performance": ["test_simple_query_latency.py", "test_complex_query_ceiling.py", "test_no_unnecessary_llm_calls.py"],
    "07_multilingual": ["test_en_fr_ar_routing.py", "test_cross_language_retrieval.py"],
    "08_storage_index": ["test_index_identity_and_ready.py", "test_cleanup_and_isolation.py"],
    "09_conversation": ["test_conversation_memory.py"],
    "10_security_privacy": ["test_security_boundaries.py"],
    "11_resilience": ["test_failure_modes.py"],
    "12_production_contracts": ["test_production_contracts.py"],
    "13_end_to_end": ["test_scenario_simple_textbook.py", "test_scenario_scanned_mixed.py", "test_scenario_multilingual.py", "test_scenario_followup_conversation.py", "test_scenario_insufficient_evidence.py", "test_scenario_large_document.py", "test_scenario_adversarial.py", "test_scenario_multi_document_library.py"],
}


@pytest.mark.high_level
def test_high_level_suite__contains_all_planned_behavior_files():
    root = Path(__file__).resolve().parent
    for directory, files in EXPECTED.items():
        actual = root / directory
        assert actual.is_dir(), directory
        for filename in files:
            assert (actual / filename).is_file(), f"missing high-level test: {directory}/{filename}"
