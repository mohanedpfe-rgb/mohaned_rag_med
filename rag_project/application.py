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
from rag_project.intelligence.med_evidence_pro import MedEvidenceProEngine
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
