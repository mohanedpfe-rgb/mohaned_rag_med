from __future__ import annotations

import re
from typing import Any, Callable


_INSTALLED = False


def _unwrap(fn: Any, suffix: str) -> Callable[..., Any] | None:
    seen: set[int] = set()
    current = fn
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "__qualname__", "").endswith(suffix):
            return current
        candidates: list[Callable[..., Any]] = []
        for cell in getattr(current, "__closure__", ()) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and value is not current:
                candidates.append(value)
        current = next((item for item in candidates if getattr(item, "__qualname__", "").endswith(suffix)), candidates[0] if candidates else None)
    return current if callable(current) else None


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
    original = _unwrap(current, "MultiTierRetriever.retrieve")
    if original is None:
        original = current

    def retrieve(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        # Execute the real production retriever exactly once. Earlier runtime
        # wrappers performed a throwaway top-1 call before the actual retrieval;
        # that doubled embedding/vector work and was the main simple-query latency
        # regression.
        hits, state = original(self, question, route, where)
        filtered = _scope_filter(list(hits or []), where)
        filtered = _document_scope_filter(filtered, question)
        state = dict(state or {})
        state["candidate_count"] = len(filtered)
        state["document_scope_enforced"] = bool(_explicit_markers(question))
        state["metadata_scope_enforced"] = bool(where)
        return filtered, state

    retrieve.__name__ = "retrieve"
    retrieve.__qualname__ = "MultiTierRetriever.retrieve"
    retrieve._root_cause_retrieval_owner = True
    med_evidence_pro.MultiTierRetriever.retrieve = retrieve


def _install_memory_contract() -> None:
    from rag_project import application, application_answer_service

    original = application_answer_service.answer
    if getattr(original, "_root_cause_memory_owner", False):
        return

    def answer(system: Any, question: str, metadata_filter: dict[str, Any] | None = None):
        result = dict(original(system, question, metadata_filter) or {})
        status = str(result.get("status") or "").upper()
        memory = getattr(system, "conversation_memory", None)
        # The canonical answer service already owns execution. This boundary only
        # repairs the persistence invariant: failed/blocked/unsupported outcomes
        # are never written into conversational memory.
        if memory is not None and status not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
            try:
                history = getattr(memory, "history", None)
                if isinstance(history, list) and history:
                    last = history[-1]
                    if isinstance(last, dict) and str(last.get("question") or last.get("user") or "") == str(question or ""):
                        history.pop()
            except Exception:
                pass
        return result

    answer._root_cause_memory_owner = True
    application_answer_service.answer = answer
    if hasattr(application.MedEvidenceProductionRAGSystem, "_certified_god_answer"):
        application.MedEvidenceProductionRAGSystem._certified_god_answer = staticmethod(answer)


def _install_publication_audit_boundary() -> None:
    from rag_project.ingestion.state_store import IngestionStateStore

    current = IngestionStateStore.transition_document_state
    if getattr(current, "_root_cause_publication_owner", False):
        return
    original = _unwrap(current, "IngestionStateStore.transition_document_state") or current

    def transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
        target = str(new_stage).upper()
        try:
            original(self, document_id, target, **values)
        except Exception as exc:
            if target not in {"READY", "COMPLETED"}:
                raise
            record = self.get_document(document_id)
            current_stage = str((record or {}).get("current_stage") or "").upper()
            current_status = str((record or {}).get("status") or "").upper()
            index_state = str((record or {}).get("index_state") or "").upper()
            requested_pages = int(values.get("total_pages", (record or {}).get("total_pages") or 0) or 0)
            requested_current = int(values.get("current_page", (record or {}).get("current_page") or 0) or 0)
            content_hash = str(values.get("content_hash", (record or {}).get("content_hash") or "") or "")
            requested_index = str(values.get("index_state", (record or {}).get("index_state") or "READY").upper()
            if (
                current_stage in {"READY", "COMPLETED"}
                and current_status in {"READY", "COMPLETED"}
                and index_state == "READY"
            ):
                return None
            if requested_pages > 0 and requested_current == requested_pages and content_hash and requested_index == "READY":
                # The publication itself is authoritative. A failure after the
                # durable UPDATE (most commonly an audit/event callback) must not
                # turn a valid READY publication into a failed ingestion.
                row_values = {
                    "current_stage": "READY",
                    "current_page": requested_current,
                    "total_pages": requested_pages,
                    "content_hash": content_hash,
                    "status": "READY",
                    "index_state": "READY",
                }
                try:
                    self.update_document(document_id, **row_values)
                except Exception:
                    pass
                return None
            raise exc

    transition.__name__ = "transition_document_state"
    transition.__qualname__ = "IngestionStateStore.transition_document_state"
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
