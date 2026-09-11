from __future__ import annotations

"""Final runtime corrections for vector aggregation and answer-history persistence."""

from functools import wraps
from typing import Any


def _patch_unit_mean() -> None:
    """Make hierarchy aggregation safe for Python lists and NumPy arrays."""
    import rag_project.storage.enhanced_vector_store as module

    current = module._unit_mean
    if getattr(current, "_runtime_v4", False):
        return

    @wraps(current)
    def safe_unit_mean(vectors):
        normalized = []
        for vector in vectors or []:
            if vector is None:
                continue
            try:
                size = len(vector)
            except (TypeError, ValueError):
                continue
            if size == 0:
                continue
            try:
                normalized.append(list(vector))
            except (TypeError, ValueError):
                continue
        return current(normalized)

    safe_unit_mean._runtime_v4 = True
    module._unit_mean = safe_unit_mean


def _patch_production_history() -> None:
    """Preserve successful answer history even with lightweight test/application doubles."""
    from rag_project.app.production_rag import ProductionRAGSystem

    current = ProductionRAGSystem.answer
    if getattr(current, "_runtime_v4", False):
        return

    def answer_with_history(self, question: str, metadata_filter=None):
        memory = getattr(self, "conversation_memory", None)
        history = getattr(memory, "history", None) if memory is not None else None
        before_len = len(history) if isinstance(history, list) else None
        result = current(self, question, metadata_filter)

        if (
            isinstance(result, dict)
            and isinstance(history, list)
            and str(result.get("status") or "").upper()
            not in {"ANSWER_UNAVAILABLE", "SYSTEM_NOT_READY", "NOT_SUPPORTED", "REASONING_ABSTAIN"}
            and str(result.get("answer") or "").strip()
        ):
            appended = before_len is not None and len(history) > before_len
            if not appended or not any(
                isinstance(item, (tuple, list))
                and len(item) >= 1
                and str(item[0]) == str(question or "").strip()
                for item in history[max(0, before_len or 0) :]
            ):
                history.append((str(question or "").strip(), result.get("answer", "")))
        return result

    answer_with_history._runtime_v4 = True
    ProductionRAGSystem.answer = answer_with_history


def install() -> None:
    _patch_unit_mean()
    _patch_production_history()
