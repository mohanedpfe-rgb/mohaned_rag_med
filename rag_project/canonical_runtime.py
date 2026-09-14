"""Single source of truth for the production answer authority."""
from __future__ import annotations

from typing import Any

ANSWER_AUTHORITY = "rag_project.application_answer_service.answer"
CANONICAL_SERVICE = "rag_project.application.MedEvidenceProductionRAGSystem"
LEGACY_ANSWER_AUTHORITY = ANSWER_AUTHORITY


def install() -> dict[str, Any]:
    """Verify canonical ownership without installing behavioral monkey patches."""
    from rag_project import application
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION
    from rag_project.runtime_invariant_repairs import install as install_invariant_repairs
    from rag_project.runtime_pdf_text_normalization import install as install_pdf_unicode_normalization

    install_pdf_unicode_normalization()
    install_invariant_repairs()

    # The legacy service remains an infrastructure compatibility adapter only.
    # Keep its exported authority marker aligned with this source of truth, but
    # never replace any legacy method implementation here.
    legacy_binding = {"module": "rag_project.app.production_rag", "authority": LEGACY_ANSWER_AUTHORITY, "role": "compatibility_only"}
    try:
        from rag_project.app import production_rag
        production_rag.ANSWER_PIPELINE_AUTHORITY = LEGACY_ANSWER_AUTHORITY
        legacy_binding["bound"] = True
    except Exception as exc:
        legacy_binding.update({"bound": False, "error": type(exc).__name__})

    service_cls = getattr(application, "MedEvidenceProductionRAGSystem", None)
    expected = getattr(application, "_med_evidence_answer", None)
    certified = getattr(service_cls, "_certified_god_answer", None) if service_cls is not None else None
    binding_installed = callable(certified) and certified is expected
    application_authority = str(getattr(application, "ACTIVE_ANSWER_PIPELINE_AUTHORITY", ""))
    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "legacy_authority": LEGACY_ANSWER_AUTHORITY,
        "legacy_binding": legacy_binding,
        "class_binding_installed": binding_installed,
        "runtime_contract_bound": binding_installed and application_authority == ANSWER_AUTHORITY,
        "health_contract_bound": callable(getattr(service_cls, "health_report", None)),
        "owner_level_repairs": True,
        "monkey_patch": False,
        "production_contract_version": CONTRACT_VERSION,
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "LEGACY_ANSWER_AUTHORITY", "install"]
