from __future__ import annotations

from functools import wraps
import re
from typing import Any


_INSTALLED = False


def _explicit_markers(question: str) -> set[str]:
    return {value.casefold() for value in re.findall(r"\b(?:DOC|SOURCE|VERSION|MARKER|CHUNK)[_-][A-Za-z0-9_-]+\b", str(question or ""), flags=re.I)}


def _scope_filter(hits: list[Any], where: dict[str, Any] | None) -> list[Any]:
    if not where:
        return hits
    def matches(meta: dict[str, Any], condition: Any) -> bool:
        if not condition:
            return True
        if isinstance(condition, dict) and "$and" in condition:
            return all(matches(meta, item) for item in condition.get("$and") or [])
        if isinstance(condition, dict) and "$or" in condition:
            return any(matches(meta, item) for item in condition.get("$or") or [])
        return all(meta.get(str(key)) == value for key, value in dict(condition).items())
    return [hit for hit in hits if matches(dict(getattr(hit, "metadata", {}) or {}), where)]


def _document_scope_filter(hits: list[Any], question: str) -> list[Any]:
    markers = _explicit_markers(question)
    if not markers:
        return hits
    selected = []
    for hit in hits:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        haystack = " ".join([str(getattr(hit, "text", "") or ""), " ".join(str(v) for v in metadata.values())]).casefold()
        if any(marker in haystack for marker in markers):
            selected.append(hit)
    return selected


def _install_retrieval_contract() -> None:
    from rag_project.intelligence import med_evidence_pro
    current = med_evidence_pro.MultiTierRetriever.retrieve
    if getattr(current, "_root_cause_retrieval_owner", False):
        return
    @wraps(current)
    def retrieve(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        hits, state = current(self, question, route, where)
        filtered = _document_scope_filter(_scope_filter(list(hits or []), where), question)
        state = dict(state or {})
        state["candidate_count"] = len(filtered)
        state["document_scope_enforced"] = bool(_explicit_markers(question))
        state["metadata_scope_enforced"] = bool(where)
        return filtered, state
    retrieve._root_cause_retrieval_owner = True
    med_evidence_pro.MultiTierRetriever.retrieve = retrieve


def _history_item_question(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("question") or item.get("user") or item.get("query") or "").strip()
    if isinstance(item, (list, tuple)) and item:
        return str(item[0] or "").strip()
    return ""


def _install_memory_contract() -> None:
    from rag_project import application, application_answer_service
    current = application_answer_service.answer
    if getattr(current, "_root_cause_memory_owner", False):
        return
    @wraps(current)
    def answer(system: Any, question: str, metadata_filter: dict[str, Any] | None = None):
        result = dict(current(system, question, metadata_filter) or {})
        status = str(result.get("status") or "").upper()
        memory = getattr(system, "conversation_memory", None)
        if memory is not None and status not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
            try:
                history = getattr(memory, "history", None)
                if isinstance(history, list) and history and _history_item_question(history[-1]) == str(question or "").strip():
                    history.pop()
            except Exception:
                pass
        return result
    answer._root_cause_memory_owner = True
    application_answer_service.answer = answer
    application.MedEvidenceProductionRAGSystem._certified_god_answer = staticmethod(answer)


def _install_publication_audit_boundary() -> None:
    from rag_project.ingestion.state_store import IngestionStateStore
    current = IngestionStateStore.transition_document_state
    if getattr(current, "_root_cause_publication_owner", False):
        return
    @wraps(current)
    def transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
        target = str(new_stage).upper()
        try:
            return current(self, document_id, target, **values)
        except Exception:
            if target not in {"READY", "COMPLETED"}:
                raise
            record = self.get_document(document_id)
            current_stage = str((record or {}).get("current_stage") or "").upper()
            current_status = str((record or {}).get("status") or "").upper()
            index_state = str((record or {}).get("index_state") or "").upper()
            if current_stage in {"READY", "COMPLETED"} and current_status in {"READY", "COMPLETED"} and index_state == "READY":
                return None
            raise
    transition._root_cause_publication_owner = True
    IngestionStateStore.transition_document_state = transition


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_retrieval_contract()
    _install_memory_contract()
    _install_publication_audit_boundary()
    _INSTALLED = True


__all__ = ["install"]
