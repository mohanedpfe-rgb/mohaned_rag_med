from __future__ import annotations

from typing import Any, Dict

from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.intelligence.god_mode import _god_answer, audit_god_mode_index
from rag_project.intelligence.final_44 import _wrap_answer
from rag_project.intelligence.medical_safety import apply_medical_safety_policy


class ProductionRAGSystem(ResilientRAGSystem):
    """Explicit production composition of retrieval, reasoning and safety policies."""

    _certified_god_answer = _wrap_answer(_god_answer)

    def answer(self, question: str, metadata_filter: Dict[str, Any] | None = None) -> dict[str, Any]:
        result = self._certified_god_answer(self, question, metadata_filter)
        return apply_medical_safety_policy(question, result, self.settings)

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
        checks["models"] = {"embedding_model": self.settings.embedding_model, "generation_model": self.settings.generation_model}
        checks["pipeline"] = {"explicit_composition": True, "answer_monkey_patch": False, "medical_safety_policy": True, "feature_count": 44}
        index_status = str(checks.get("index", {}).get("status", "READY")).upper()
        checks["ready"] = bool(checks["embedding"]["ok"] and index_status in {"READY", "OK"})
        return checks


__all__ = ["ProductionRAGSystem"]
