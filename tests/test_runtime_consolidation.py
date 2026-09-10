from __future__ import annotations

from rag_project.application import runtime_contract


def test_runtime_contract_exposes_one_canonical_production_path() -> None:
    contract = runtime_contract()
    assert contract["composition_root"] == "rag_project.application.create_rag_system"
    assert contract["canonical_service"] == "rag_project.app.production_rag.ProductionRAGSystem"
    assert contract["service"] == "ProductionRAGSystem"
    assert contract["canonical_ingestion"] == "rag_project.ingestion.robust_ingestor.robust_ingest_file"
    assert contract["runtime_policy"] == "rag_project.runtime.install"
    assert contract["storage_policy"] == "rag_project.storage.vector_store_runtime.install"
    assert "authentication_policy" not in contract
    assert contract["answer_monkey_patch"] is False
