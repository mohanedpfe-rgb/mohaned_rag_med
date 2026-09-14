"""Single source of truth for the production answer authority."""
from __future__ import annotations

from typing import Any

ANSWER_AUTHORITY = "rag_project.intelligence.top_level_pipeline.complete_phases"
CANONICAL_SERVICE = "rag_project.application.MedEvidenceProductionRAGSystem"
LEGACY_ANSWER_AUTHORITY = ANSWER_AUTHORITY


def install() -> dict[str, Any]:
    """Validate canonical ownership without installing behavioral monkey patches.

    The composition root may validate contracts here, but it must not mutate the
    legacy implementation or install runtime wrappers.  The legacy service is
    reachable only through the explicit compatibility adapter.
    """
    from rag_project import application
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION

    service_cls = getattr(application, "MedEvidenceProductionRAGSystem", None)
    expected = getattr(application, "_med_evidence_answer", None)
    certified = getattr(service_cls, "_certified_god_answer", None) if service_cls is not None else None
    binding_installed = callable(certified) and certified is expected
    application_authority = str(getattr(application, "ACTIVE_ANSWER_PIPELINE_AUTHORITY", ""))

    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "legacy_authority": LEGACY_ANSWER_AUTHORITY,
        "legacy_binding": {"role": "compatibility_adapter_only", "patched": False},
        "class_binding_installed": binding_installed,
        "runtime_contract_bound": binding_installed and application_authority == ANSWER_AUTHORITY,
        "health_contract_bound": callable(getattr(service_cls, "health_report", None)),
        "owner_level_repairs": True,
        "monkey_patch": False,
        "production_contract_version": CONTRACT_VERSION,
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "LEGACY_ANSWER_AUTHORITY", "install"]
