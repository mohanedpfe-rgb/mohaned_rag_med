from __future__ import annotations

"""Final runtime corrections for answer-history persistence and defensive compatibility."""


def _patch_production_history() -> None:
    """Persist successful answers when a lightweight memory double lacks add()."""
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
            new_items = history[max(0, before_len or 0) :] if before_len is not None else []
            already_recorded = any(
                isinstance(item, (tuple, list))
                and len(item) >= 1
                and str(item[0]) == str(question or "").strip()
                for item in new_items
            )
            if not already_recorded:
                history.append((str(question or "").strip(), str(result.get("answer") or "")))
        return result

    answer_with_history._runtime_v4 = True
    ProductionRAGSystem.answer = answer_with_history


def install() -> None:
    _patch_production_history()
