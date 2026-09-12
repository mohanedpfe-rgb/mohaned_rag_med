"""Single source of truth for the production answer authority."""
from __future__ import annotations

from typing import Any

ANSWER_AUTHORITY = "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer"
CANONICAL_SERVICE = "rag_project.application.MedEvidenceProductionRAGSystem"


def install() -> dict[str, Any]:
    """Inspect and return the canonical runtime contract without mutating modules."""
    from rag_project import application
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION

    service_cls = getattr(application, "MedEvidenceProductionRAGSystem", None)
    certified = getattr(service_cls, "_certified_god_answer", None) if service_cls is not None else None
    expected = getattr(application, "_med_evidence_answer", None)
    binding_installed = callable(certified) and callable(expected) and certified is expected
    authority = str(getattr(application, "ACTIVE_ANSWER_PIPELINE_AUTHORITY", ""))
    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "class_binding_installed": binding_installed,
        "runtime_contract_bound": binding_installed and authority == ANSWER_AUTHORITY,
        "health_contract_bound": callable(getattr(service_cls, "health_report", None)),
        "monkey_patch": False,
        "production_contract_version": CONTRACT_VERSION,
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "install"]
