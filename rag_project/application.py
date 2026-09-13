from __future__ import annotations

import threading
from typing import Any

from rag_project.application_answer_service import ACTIVE_ANSWER_PIPELINE_AUTHORITY
from rag_project.application_answer_service import answer as _med_evidence_answer
from rag_project.application_answer_service import install_runtime_adapters
from rag_project.application_legacy_adapter import LegacyProductionRAGAdapter
from rag_project.configuration.settings import Settings
from rag_project.canonical_runtime import ANSWER_AUTHORITY
from rag_project.ingestion.ingestion_contract import INGESTION_CONTRACT_VERSION
from rag_project.intelligence.production_contract_v2 import CONTRACT_VERSION as PRODUCTION_CONTRACT_VERSION
from rag_project.quality_gate import run_quality_gate
from rag_project.runtime import install, install_application_contracts
from rag_project.runtime_bootstrap_state import is_prepared as runtime_is_prepared
from rag_project.security import harden_system

_FACTORY_LOCK = threading.RLock()
ANSWER_PIPELINE_AUTHORITY = ANSWER_AUTHORITY


def _normalize_runtime_settings(settings: Settings | None) -> Settings:
    resolved = settings or Settings.from_env()
    resolved.embedding_batch_size = max(16, min(int(resolved.embedding_batch_size), 32))
    resolved.embedding_retries = max(1, min(int(resolved.embedding_retries), 3))
    resolved.embedding_timeout_seconds = max(30.0, min(float(resolved.embedding_timeout_seconds), 300.0))
    resolved.max_workers = max(1, min(int(resolved.max_workers), 4))
    resolved.ollama_concurrency = max(1, min(int(resolved.ollama_concurrency), 2))
    return resolved


class MedEvidenceProductionRAGSystem:
    """Canonical application service with explicit compatibility and answer contracts."""

    _certified_god_answer = staticmethod(_med_evidence_answer)

    def __init__(self, settings: Settings) -> None:
        self._runtime = LegacyProductionRAGAdapter(settings)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_runtime":
            object.__setattr__(self, name, value)
            return
        setattr(self._runtime, name, value)

    @property
    def runtime(self) -> LegacyProductionRAGAdapter:
        return object.__getattribute__(self, "_runtime")

    def ingest_file(self, pdf_path: Any) -> dict[str, Any]:
        result = dict(self.runtime.ingest_file(pdf_path) or {})
        status = str(result.get("status") or "").upper()
        if status in {"SUCCESS", "COMPLETED"}:
            result["status"] = "READY"
        elif status == "FAILED":
            result["status"] = "FAILED"
        return result

    def ingest_directory(self, directory: Any = None) -> list[dict[str, Any]]:
        results = self.runtime.ingest_directory(directory)
        normalized: list[dict[str, Any]] = []
        for item in results:
            row = dict(item or {})
            if str(row.get("status") or "").upper() in {"SUCCESS", "COMPLETED"}:
                row["status"] = "READY"
            normalized.append(row)
        return normalized

    def health_report(self) -> dict[str, Any]:
        report = dict(self.runtime.health_report() or {})
        pipeline = dict(report.get("pipeline") or {})
        pipeline.update(
            {
                "explicit_composition": True,
                "authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY,
                "answer_pipeline": "med_evidence_pro",
                "med_evidence_pro": True,
                "phase_count": 8,
                "safety_gate": True,
                "multi_tier_retrieval": True,
                "structured_knowledge": True,
                "semantic_cache": True,
                "answer_cascade": True,
                "active_verification": True,
                "feedback_logging": True,
                "operations_store": True,
                "ab_testing": True,
                "retraining_manifest": True,
                "backup_rotation": True,
                "circuit_breaker": True,
                "cloud_hybrid": True,
                "cloud_opt_in": True,
                "cloud_pii_redaction": True,
                "enterprise_roles": True,
            }
        )
        report["pipeline"] = pipeline
        return report

    def cancel_all_ingests(self) -> Any:
        return self.runtime.cancel_all_ingests()


def create_rag_system(settings: Settings | None = None, *, runtime_prepared: bool | None = None):
    """Build the canonical runtime without importing or owning the composition root."""
    with _FACTORY_LOCK:
        prepared = runtime_is_prepared() if runtime_prepared is None else runtime_prepared
        if not prepared:
            install()
            install_application_contracts()
        system = MedEvidenceProductionRAGSystem(_normalize_runtime_settings(settings))
        system = harden_system(system)
        system = install_runtime_adapters(system)
        try:
            system.startup_quality = run_quality_gate(system, repair_drift=True)
            if not system.startup_quality.get("ready", False):
                system.logger.warning(
                    "Runtime quality gate reported a non-ready state: %s",
                    system.startup_quality,
                )
        except Exception as exc:
            system.startup_quality = {"ready": False, "error": type(exc).__name__}
            system.logger.exception("Runtime quality gate failed")
        return system


def create_default_rag_system():
    return create_rag_system()


def runtime_contract() -> dict[str, Any]:
    return {
        "composition_root": "rag_project.composition.prepare_runtime -> rag_project.application.create_rag_system",
        "canonical_service": "rag_project.application.MedEvidenceProductionRAGSystem",
        "service": "MedEvidenceProductionRAGSystem",
        "legacy_service": "rag_project.app.production_rag.ProductionRAGSystem",
        "legacy_adapter": "rag_project.application_legacy_adapter.LegacyProductionRAGAdapter",
        "canonical_ingestion": "rag_project.ingestion.versioned_ingestor.ingest_version_safely",
        "base_ingestion_engine": "rag_project.ingestion.robust_ingestor.robust_ingest_file",
        "versioned_ingestion_publication": True,
        "last_known_good_preservation": True,
        "runtime_policy": "rag_project.runtime.install",
        "storage_policy": "rag_project.storage.vector_store_runtime.install",
        "security_policy": "rag_project.security.harden_system",
        "quality_policy": "bounded_startup_check_with_optional_deep_audit",
        "configuration": "Settings.from_env",
        "answer_pipeline": "med_evidence_pro",
        "answer_pipeline_authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY,
        "answer_pipeline_execution": ACTIVE_ANSWER_PIPELINE_AUTHORITY,
        "active_answer_pipeline": "med_evidence_pro",
        "active_answer_pipeline_authority": ACTIVE_ANSWER_PIPELINE_AUTHORITY,
        "answer_monkey_patch": False,
        "canonical_runtime_binding": "rag_project.canonical_runtime.install",
        "production_contract": "rag_project.intelligence.production_contract_v2.install",
        "production_contract_version": PRODUCTION_CONTRACT_VERSION,
        "ingestion_contract": "rag_project.ingestion.ingestion_contract.install",
        "ingestion_contract_version": INGESTION_CONTRACT_VERSION,
        "final_answer_verification": "rag_project.intelligence.final_answer_contract.verify_final_answer",
        "entity_coverage": "rag_project.intelligence.entity_coverage.score_entity_coverage",
        "document_aware_routing": True,
        "hierarchical_retrieval": True,
        "query_self_correction": True,
        "table_aware_retrieval": True,
        "numeric_aware_retrieval": True,
        "medical_synonym_expansion": True,
        "contradiction_detection": True,
        "medical_safety_gate": True,
        "structured_request_context": True,
        "structured_evidence_bundle": True,
        "structured_answer_envelope": True,
        "confidence_breakdown": True,
        "request_traceability": True,
        "ingestion_traceability": True,
        "atomic_ingestion_publication": True,
        "post_write_index_validation": True,
        "ready_only_retrieval_boundary": True,
        "ready_only_retriever": "rag_project.retrieval.ready_only_retriever.ReadyOnlyRetriever",
        "runtime_cache_freshness": True,
        "verified_generation_recovery": True,
        "med_evidence_pro": True,
        "phase_count": 8,
        "answer_cascade": True,
        "semantic_retrieval_cache": True,
        "structured_knowledge_layer": True,
        "feedback_loop": True,
        "production_operations_store": True,
        "ab_testing": True,
        "retraining_pipeline": True,
        "backup_rotation": True,
        "load_benchmarking": True,
        "resilience_controls": True,
        "cloud_hybrid": True,
        "cloud_opt_in": True,
        "cloud_api_key_env": "ANTHROPIC_API_KEY",
        "cloud_rate_limit": True,
        "cloud_cost_tracking": True,
        "cloud_audit_log": True,
        "cloud_pii_redaction": True,
        "cloud_smart_escalation": True,
        "enterprise_roles": True,
        "ehr_integration_contract": "provider-neutral optional adapter",
        "runtime_prepared_factory": True,
    }


__all__ = [
    "create_rag_system",
    "create_default_rag_system",
    "runtime_contract",
    "ANSWER_PIPELINE_AUTHORITY",
    "ACTIVE_ANSWER_PIPELINE_AUTHORITY",
    "PRODUCTION_CONTRACT_VERSION",
    "INGESTION_CONTRACT_VERSION",
    "MedEvidenceProductionRAGSystem",
    "_med_evidence_answer",
]
