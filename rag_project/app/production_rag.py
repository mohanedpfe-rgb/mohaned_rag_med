from __future__ import annotations

from typing import Any, Dict, List

from rag_project.app import rag_system as rag_system_module
from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.ingestion import robust_ingestor
from rag_project.intelligence.god_mode import _god_answer, audit_god_mode_index
from rag_project.intelligence.final_44 import _wrap_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy
from rag_project.intelligence.production_contract import (
    sanitize_trace,
    validate_feature_contract,
)


class ProductionRAGSystem(ResilientRAGSystem):
    """Explicit production composition with resilient incremental PDF ingestion."""

    _certified_god_answer = _wrap_answer(_god_answer)

    def __init__(self, settings: Any | None = None):
        super().__init__(settings)
        self._production_feature_contract = validate_feature_contract()

    def _new_cancel_flag(self, document_id: str):
        flag = rag_system_module._IngestCancelFlag()
        with rag_system_module._INGEST_LOCK:
            rag_system_module._INGEST_CANCEL_FLAGS[document_id] = flag
        return flag

    def _remove_cancel_flag(self, document_id: str) -> None:
        with rag_system_module._INGEST_LOCK:
            rag_system_module._INGEST_CANCEL_FLAGS.pop(document_id, None)

    def ingest_file(self, pdf_path: str | Any) -> dict[str, Any]:
        """Use incremental ingestion and fail fast when embeddings are unavailable."""
        self.embedding_startup_error = None
        self._ensure_embedding_dimension()
        if self.embedding_startup_error:
            return {
                "status": "failed",
                "file_name": getattr(pdf_path, "name", str(pdf_path)),
                "error": f"FAILED_EMBEDDING: {self.embedding_startup_error}",
            }
        return robust_ingestor.robust_ingest_file(self, pdf_path)

    def ingest_directory(self, directory: str | Any | None = None) -> List[Dict[str, Any]]:
        """Process PDFs serially in production to avoid Chroma/SQLite contention.

        The ingestion pipeline is already incremental, so serial execution keeps
        the shared local vector database responsive while still exposing live
        page/batch progress for the active document.
        """
        from pathlib import Path

        source_dir = Path(directory) if directory else self.settings.incoming_dir
        source_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        for pdf_path in sorted(source_dir.glob("*.pdf")):
            result = self.ingest_file(pdf_path)
            results.append(result)
        return results

    def cancel_all_ingests(self) -> int:
        """Cancel ingestion jobs owned by the shared ingestion registry."""
        with rag_system_module._INGEST_LOCK:
            count = 0
            for flag in rag_system_module._INGEST_CANCEL_FLAGS.values():
                if not flag.cancelled:
                    flag.cancel()
                    count += 1
            return count

    def answer(self, question: str, metadata_filter: Dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._production_feature_contract["all_resolved"]:
            return {
                "status": "SYSTEM_NOT_READY",
                "answer": "The production feature contract is incomplete; a grounded answer is disabled.",
                "citations": [],
                "hits": [],
                "confidence": {"level": "none", "evidence_confidence": 0.0},
                "production_contract": self._production_feature_contract,
            }
        result = self._certified_god_answer(self, question, metadata_filter)
        result = apply_medical_safety_policy(question, result, self.settings)
        if "query_trace" in result:
            result["query_trace"] = sanitize_trace(result["query_trace"])
        result["production_contract"] = {"feature_count": 44, "all_features_resolved": True}
        return result

    def audit_god_mode_index(self) -> dict[str, Any]:
        return audit_god_mode_index(self)

    def health_report(self) -> dict[str, Any]:
        """Return deterministic readiness information without mutating index data."""
        checks: dict[str, Any] = {}
        try:
            self._ensure_embedding_dimension()
            checks["embedding"] = {"ok": True, "identity": self.embedding_service.identity}
        except Exception as exc:
            checks["embedding"] = {"ok": False, "error": str(exc)}
        try:
            checks["index"] = self.vector_store.compatibility_report(self.embedding_service.identity)
        except Exception as exc:
            checks["index"] = {"status": "UNAVAILABLE", "error": str(exc)}
        try:
            checks["audit"] = self.audit_god_mode_index()
        except Exception as exc:
            checks["audit"] = {"ok": False, "error": str(exc)}
        checks["feature_contract"] = self._production_feature_contract
        checks["models"] = {
            "embedding_model": self.settings.embedding_model,
            "generation_model": self.settings.generation_model,
        }
        checks["pipeline"] = {
            "explicit_composition": True,
            "answer_monkey_patch": False,
            "medical_safety_policy": True,
            "privacy_safe_trace": True,
            "feature_count": 44,
            "incremental_ingestion": True,
            "bounded_embedding_commit": True,
            "durable_page_checkpoints": True,
            "serialized_local_index_writes": True,
        }
        index_status = str(checks.get("index", {}).get("status", "READY")).upper()
        checks["ready"] = bool(
            checks["embedding"]["ok"]
            and index_status in {"READY", "OK"}
            and self._production_feature_contract["all_resolved"]
        )
        return checks


__all__ = ["ProductionRAGSystem"]
