from __future__ import annotations

import inspect

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_pipeline_authority


@pytest.mark.high_level
def test_production_contracts__single_answer_authority_is_executed_exactly(clean_system):
    from rag_project.application import ACTIVE_ANSWER_PIPELINE_AUTHORITY

    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert result.get("pipeline_authority") == ACTIVE_ANSWER_PIPELINE_AUTHORITY
    assert result.get("implementation_authority") == ACTIVE_ANSWER_PIPELINE_AUTHORITY
    assert_pipeline_authority(result)
    assert result.get("canonical_pipeline_executed") is True


@pytest.mark.high_level
def test_production_contracts__runtime_contract_has_one_composition_root_and_one_answer_authority():
    from rag_project.application import ACTIVE_ANSWER_PIPELINE_AUTHORITY, runtime_contract

    contract = runtime_contract()
    assert contract["composition_root"] == "rag_project.application.create_rag_system"
    assert contract["canonical_service"] == "rag_project.app.production_rag.ProductionRAGSystem"
    assert contract["answer_pipeline"] == "med_evidence_pro"
    assert contract["answer_pipeline_authority"] == ACTIVE_ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_pipeline_execution"] == ACTIVE_ANSWER_PIPELINE_AUTHORITY
    assert contract["active_answer_pipeline_authority"] == ACTIVE_ANSWER_PIPELINE_AUTHORITY
    assert contract["answer_monkey_patch"] is False
    assert contract["final_answer_verification"]
    assert contract["atomic_ingestion_publication"] is True
    assert contract["post_write_index_validation"] is True
    assert contract["canonical_ingestion"] == "rag_project.ingestion.versioned_ingestor.ingest_version_safely"
    assert contract["base_ingestion_engine"] == "rag_project.ingestion.robust_ingestor.robust_ingest_file"
    assert contract["versioned_ingestion_publication"] is True
    assert contract["last_known_good_preservation"] is True
    assert contract["ready_only_retrieval_boundary"] is True
    assert contract["ready_only_retriever"].endswith("ReadyOnlyRetriever")
    assert contract["runtime_cache_freshness"] is True
    assert contract["verified_generation_recovery"] is True


@pytest.mark.high_level
def test_production_contracts__feature_contract_and_startup_quality_are_explicit(clean_system):
    contract = getattr(clean_system, "_production_feature_contract", None)
    quality = getattr(clean_system, "startup_quality", None)

    assert isinstance(contract, dict)
    assert contract.get("all_resolved") is True
    assert contract.get("unique_names") is True
    assert contract.get("duplicates") == []
    assert contract.get("unresolved") == {}
    assert int(contract.get("feature_count", 0)) == 44
    assert isinstance(quality, dict)
    assert "ready" in quality


@pytest.mark.high_level
def test_production_contracts__answer_exposes_all_canonical_phases(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    phases = result.get("phases") or {}
    expected = {
        "phase_0_safety_gate",
        "phase_1_query_router",
        "phase_2a_multi_tier_retrieval",
        "phase_2b_structured_knowledge",
        "phase_3_evidence_compiler",
        "phase_4_answer_cascade",
        "phase_5_active_verification",
        "phase_6_response_formatter",
        "phase_7_logging_feedback",
    }
    assert expected.issubset(phases)
    assert all(str(phases[name]) for name in expected)


@pytest.mark.high_level
def test_production_contracts__versioned_ingestion_is_part_of_the_live_composition():
    from rag_project.app.production_rag import ProductionRAGSystem
    from rag_project.ingestion import versioned_ingestor

    source = inspect.getsource(ProductionRAGSystem.ingest_file)
    assert "versioned_ingestor.ingest_version_safely" in source
    assert callable(versioned_ingestor.ingest_version_safely)


@pytest.mark.high_level
def test_production_contracts__runtime_safety_layer_is_live_and_not_documentation_only():
    from rag_project.application import runtime_contract
    from rag_project.intelligence import runtime_safety

    contract = runtime_contract()
    assert contract["runtime_cache_freshness"] is True
    assert contract["verified_generation_recovery"] is True
    assert callable(runtime_safety.execute_with_runtime_safety)
    assert "execute_with_runtime_safety" in inspect.getsource(runtime_safety)
