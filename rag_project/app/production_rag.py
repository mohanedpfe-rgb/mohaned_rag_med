from __future__ import annotations

from typing import Any, Dict

from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.intelligence.god_mode import run_god_mode
from rag_project.intelligence.final_44 import certify_result
from rag_project.intelligence.medical_safety import apply_medical_safety_policy


class ProductionRAGSystem(ResilientRAGSystem):
    """Explicit production composition of retrieval, reasoning and safety policies.

    This class intentionally delegates instead of monkey-patching RAGSystem.answer.
    The pipeline is therefore visible from the composition root and testable as a
    normal Python call graph.
    """

    def answer(self, question: str, metadata_filter: Dict[str, Any] | None = None) -> dict[str, Any]:
        result = run_god_mode(self, question, metadata_filter)
        result = certify_result(self, question, result)
        return apply_medical_safety_policy(question, result, self.settings)

    def health_report(self) -> dict[str, Any]:
        """Return a deterministic readiness report without mutating application state."""
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
        checks["models"] = {
            "embedding_model": self.settings.embedding_model,
            "generation_model": self.settings.generation_model,
        }
        checks["pipeline"] = {
            "explicit_composition": True,
            "monkey_patch_answer": False,
            "medical_safety_policy": True,
            "feature_count": 44,
        }
        index_ok = str(checks.get("index", {}).get("status", "READY")).upper() in {"READY", "OK"}
        checks["ready"] = bool(checks["embedding"]["ok"] and index_ok)
        return checks


__all__ = ["ProductionRAGSystem"]
