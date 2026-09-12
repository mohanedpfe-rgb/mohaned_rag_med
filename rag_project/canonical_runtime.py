"""Single source of truth for the production answer authority."""
from __future__ import annotations

from typing import Any

ANSWER_AUTHORITY = "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer"
CANONICAL_SERVICE = "rag_project.app.production_rag.ProductionRAGSystem"


def install() -> dict[str, Any]:
    """Return the canonical runtime contract without mutating imported modules."""
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION

    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "class_binding_installed": False,
        "runtime_contract_bound": True,
        "health_contract_bound": True,
        "monkey_patch": False,
        "production_contract_version": CONTRACT_VERSION,
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "install"]
