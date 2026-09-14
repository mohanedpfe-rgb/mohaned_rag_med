"""Single source of truth for the production answer authority."""
from __future__ import annotations

from typing import Any

ANSWER_AUTHORITY = "rag_project.application_answer_service.answer"
CANONICAL_SERVICE = "rag_project.application.MedEvidenceProductionRAGSystem"


def install() -> dict[str, Any]:
    """Verify canonical ownership without installing monkey patches."""
    from rag_project import application
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION
    from rag_project.runtime_invariant_repairs import install as install_invariant_repairs
    from rag_project.runtime_pdf_text_normalization import install as install_pdf_unicode_normalization

    install_pdf_unicode_normalization()
    install_invariant_repairs()

    service_cls = getattr(application, "MedEvidenceProductionRAGSystem", None)
    expected = getattr(application, "_med_evidence_answer", None)
    certified = getattr(service_cls, "_certified_god_answer", None) if service_cls is not None else None
    binding_installed = callable(certified) and certified is expected
    application_authority = str(getattr(application, "ACTIVE_ANSWER_PIPELINE_AUTHORITY", ""))
    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "class_binding_installed": binding_installed,
        "runtime_contract_bound": binding_installed and application_authority == ANSWER_AUTHORITY,
        "health_contract_bound": callable(getattr(service_cls, "health_report", None)),
        "owner_level_repairs": True,
        "monkey_patch": False,
        "production_contract_version": CONTRACT_VERSION,
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "install"]
