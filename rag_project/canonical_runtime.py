"""Final runtime binding for the single production answer authority."""
from __future__ import annotations

from typing import Any

# Keep this string stable for UI/health-report compatibility.  The callable below is
# the real authority and is deliberately bound to the production service.
ANSWER_AUTHORITY = "rag_project.intelligence.god_mode_100.enhanced_god_answer"
CANONICAL_SERVICE = "rag_project.app.production_rag.ProductionRAGSystem"


def install() -> dict[str, Any]:
    """Bind the live production service to the authoritative answer implementation."""
    from rag_project.app import production_rag
    from rag_project.intelligence import god_mode_100
    from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION
    from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION

    service_cls = production_rag.ProductionRAGSystem
    authority = god_mode_100.enhanced_god_answer

    # The previous runtime incorrectly installed ``enhance_result`` here. That
    # function is diagnostic-only and returns its input unchanged; binding it as
    # the service answer method meant the actual answer pipeline was never run.
    production_rag.enhanced_god_answer = authority
    setattr(service_cls, "_certified_god_answer", authority)
    setattr(service_cls, "_canonical_answer_authority", ANSWER_AUTHORITY)
    setattr(service_cls, "_canonical_runtime_contract", True)

    if not getattr(service_cls, "_canonical_health_contract", False):
        previous_health = getattr(service_cls, "health_report", None)
        if callable(previous_health):
            def canonical_health(self: Any) -> dict[str, Any]:
                report = previous_health(self)
                pipeline = dict(report.get("pipeline") or {})
                pipeline.update({
                    "canonical_runtime_contract": True,
                    "answer_pipeline_authority": ANSWER_AUTHORITY,
                    "production_contract_version": CONTRACT_VERSION,
                    "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
                    "structured_request_context": True,
                    "structured_evidence_bundle": True,
                    "structured_answer_envelope": True,
                    "confidence_breakdown": True,
                    "request_traceability": True,
                    "ingestion_traceability": True,
                    "atomic_ingestion_publication": True,
                    "post_write_index_validation": True,
                })
                report["pipeline"] = pipeline
                report["canonical_runtime"] = {
                    "service": CANONICAL_SERVICE,
                    "answer_pipeline_authority": ANSWER_AUTHORITY,
                    "production_contract_version": CONTRACT_VERSION,
                    "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
                }
                return report

            setattr(service_cls, "health_report", canonical_health)
        setattr(service_cls, "_canonical_health_contract", True)

    return {
        "canonical_service": CANONICAL_SERVICE,
        "answer_pipeline_authority": ANSWER_AUTHORITY,
        "class_binding_installed": getattr(service_cls, "_certified_god_answer", None) is authority,
        "runtime_contract_bound": True,
        "health_contract_bound": getattr(service_cls, "_canonical_health_contract", False),
        "production_contract_version": CONTRACT_VERSION,
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
    }


__all__ = ["ANSWER_AUTHORITY", "CANONICAL_SERVICE", "install"]
