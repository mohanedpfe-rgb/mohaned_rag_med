from __future__ import annotations

from typing import Any

_INSTALLED = False


def _install_production_history_contract() -> None:
    from rag_project.app.production_rag import ProductionRAGSystem, _should_store_in_history
    current = getattr(ProductionRAGSystem, "answer", None)
    if not callable(current) or getattr(current, "_runtime_v8_history", False): return

    def answer(self, question: str, metadata_filter=None):
        result = current(self, question, metadata_filter)
        memory = getattr(self, "conversation_memory", None)
        if memory is None or not _should_store_in_history(result): return result
        add = getattr(memory, "add", None)
        if callable(add): return result
        history = getattr(memory, "history", None)
        original = str(question or "").strip()
        if isinstance(history, list) and original and (not history or history[-1][0] != original):
            history.append((original, str(result.get("answer", ""))))
            max_history = getattr(memory, "max_history", None)
            if isinstance(max_history, int) and max_history > 0 and len(history) > max_history:
                del history[:-max_history]
        return result

    answer._runtime_v8_history = True
    ProductionRAGSystem.answer = answer


def _install_god_mode_contract() -> None:
    import rag_project.intelligence.god_mode_100 as module

    def enhance_result(system: Any, question: str, base_result: Any, metadata_filter=None):
        base = dict(base_result or {})
        complete = getattr(module, "complete_phases", None)
        if callable(complete):
            try:
                completed = complete(system, question, base, metadata_filter)
                if isinstance(completed, dict):
                    base = completed
            except Exception:
                pass
        enhancer = getattr(module, "_diagnostic_enhance", None)
        return enhancer(system, question, base, metadata_filter) if callable(enhancer) else base

    enhance_result._runtime_v8 = True
    module.enhance_result = enhance_result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    # Follow-up and numeric compatibility must remain at their authoritative
    # module boundaries. Previous v8 versions replaced function __code__ objects
    # across modules, which left those functions executing with incompatible
    # globals and caused NameError failures. We intentionally do not graft code
    # across modules here.
    _install_production_history_contract()
    _install_god_mode_contract()
    _INSTALLED = True
