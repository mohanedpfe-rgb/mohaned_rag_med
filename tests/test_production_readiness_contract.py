from rag_project.application import runtime_contract
from rag_project.intelligence.production_contract import validate_feature_contract
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION, install as install_production_contract
from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION
from rag_project.canonical_runtime import ANSWER_AUTHORITY, install as install_canonical_runtime
from rag_project.intelligence import god_mode_100, top_level_pipeline
from rag_project.intelligence.pipeline_integrity import install as install_pipeline_integrity
from rag_project.app.production_rag import ProductionRAGSystem


def _install_all_runtime_contracts():
    install_pipeline_integrity()
    install_production_contract()
    install_canonical_runtime()


def test_feature_contract_resolves_all_advertised_capabilities():
    report = validate_feature_contract()
    assert report["feature_count"] == 44
    assert report["unique_names"] is True
    assert report["all_resolved"] is True, report


def test_runtime_contract_declares_single_canonical_authority():
    contract = runtime_contract()
    assert contract["answer_pipeline_authority"] == ANSWER_AUTHORITY
    assert contract["answer_pipeline_execution"] == contract["answer_pipeline_authority"]
    assert contract["answer_monkey_patch"] is False
    assert contract["structured_request_context"] is True
    assert contract["structured_evidence_bundle"] is True
    assert contract["structured_answer_envelope"] is True
    assert contract["confidence_breakdown"] is True
    assert contract["ingestion_traceability"] is True
    assert contract["production_contract_version"] == CONTRACT_VERSION
    assert contract["ingestion_contract_version"] == INGESTION_CONTRACT_VERSION


def test_live_production_service_is_bound_to_installed_canonical_enhancer():
    _install_all_runtime_contracts()
    assert ProductionRAGSystem._certified_god_answer is god_mode_100.enhance_result
    assert ProductionRAGSystem._canonical_answer_authority == ANSWER_AUTHORITY
    assert ProductionRAGSystem._canonical_runtime_contract is True
    assert ProductionRAGSystem._canonical_health_contract is True


def test_live_rewrite_does_not_pollute_standalone_question():
    _install_all_runtime_contracts()
    rewritten = top_level_pipeline.rewrite_follow_up(
        "What are the main findings?",
        (("What are the complications of diabetes?", "The answer discussed diabetic nephropathy."),),
    )
    assert rewritten == "What are the main findings?"
    assert "relevant entities:" not in rewritten.casefold()
    assert "follow-up:" not in rewritten.casefold()


def test_health_contract_is_marked_installed():
    _install_all_runtime_contracts()
    assert getattr(ProductionRAGSystem, "_canonical_health_contract", False) is True
