from rag_project.application import runtime_contract
from rag_project.intelligence.production_contract import validate_feature_contract
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION


def test_feature_contract_resolves_all_advertised_capabilities():
    report = validate_feature_contract()
    assert report["feature_count"] == 44
    assert report["unique_names"] is True
    assert report["all_resolved"] is True, report


def test_runtime_contract_declares_single_canonical_authority():
    contract = runtime_contract()
    assert contract["answer_pipeline_authority"] == "rag_project.intelligence.top_level_pipeline.complete_phases"
    assert contract["answer_pipeline_execution"] == contract["answer_pipeline_authority"]
    assert contract["answer_monkey_patch"] is False
    assert contract["structured_request_context"] is True
    assert contract["structured_evidence_bundle"] is True
    assert contract["structured_answer_envelope"] is True
    assert contract["confidence_breakdown"] is True
    assert contract["ingestion_traceability"] is True
    assert contract["production_contract_version"] == CONTRACT_VERSION.removesuffix("-v2") + "-v2"
    assert contract["ingestion_contract_version"] == INGESTION_CONTRACT_VERSION
