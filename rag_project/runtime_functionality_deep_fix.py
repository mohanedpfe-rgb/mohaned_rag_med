"""Final functionality-only repairs for deterministic answer behavior."""
from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False
_TLS = threading.local()


def _wrap_retrieval_cache_fallthrough(original):
    """A cache hit that becomes irrelevant after scope filtering must not become a false abstention."""
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        hits, state = original(self, question, route, where)
        state = dict(state or {})
        if where is None and str(state.get("tier", "")).upper() == "CACHE" and not hits:
            cache = getattr(self, "cache", None)
            delete = getattr(cache, "delete", None)
            if callable(delete):
                delete(question)
            return original(self, question, route, where)
        return hits, state
    wrapped._functionality_cache_fallthrough = True
    return wrapped


def _wrap_retrieval_skip_health_probe(original):
    """Do not let the deep runtime's unused top_k=1 health probe break the real retrieval path."""
    def wrapped(self: Any, question: str, route: Any, where: dict[str, Any] | None = None):
        retriever = getattr(getattr(self, "system", None), "retriever", None)
        method = getattr(retriever, "retrieve", None)
        if not callable(method):
            return original(self, question, route, where)

        called = {"probe": False}
        original_method = method

        def proxy(query: Any, top_k: Any = 8, filter_where: Any = None, *args: Any, **kwargs: Any):
            if (
                not called["probe"]
                and query == question
                and top_k == 1
                and filter_where == where
                and not args
                and not kwargs
            ):
                called["probe"] = True
                return []
            return original_method(query, top_k, filter_where, *args, **kwargs)

        try:
            retriever.retrieve = proxy
            return original(self, question, route, where)
        finally:
            retriever.retrieve = original_method
    wrapped._functionality_probe_guard = True
    return wrapped


def _citation_complete_without_shared_state(original):
    """Validate citations against evidence actually supplied to the current generation."""
    def wrapped(answer: str, hit_count: int) -> bool:
        expected = getattr(_TLS, "citation_limit", None)
        return original(answer, int(expected if expected is not None else hit_count))
    wrapped._functionality_citation_guard = True
    return wrapped


def _wrap_generate(original):
    def wrapped(self: Any, question: str, route: Any, compiled: dict[str, Any]):
        claims = list(compiled.get("claims") or [])
        source_numbers = [
            int(number)
            for claim in claims
            for number in (getattr(claim, "source_numbers", ()) or ())
            if isinstance(number, int) or str(number).isdigit()
        ]
        previous = getattr(_TLS, "citation_limit", None)
        _TLS.citation_limit = max(source_numbers, default=0)
        try:
            return original(self, question, route, compiled)
        finally:
            if previous is None:
                try:
                    del _TLS.citation_limit
                except AttributeError:
                    pass
            else:
                _TLS.citation_limit = previous
    wrapped._functionality_generate_guard = True
    return wrapped


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.intelligence import med_evidence_pro

        original_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(original_retrieve, "_functionality_probe_guard", False):
            # Put the no-op probe guard closest to the implementation so it only suppresses
            # the artificial top_k=1 call and leaves the real retrieval call untouched.
            original_retrieve = _wrap_retrieval_skip_health_probe(original_retrieve)
            med_evidence_pro.MultiTierRetriever.retrieve = original_retrieve

        current_retrieve = med_evidence_pro.MultiTierRetriever.retrieve
        if not getattr(current_retrieve, "_functionality_cache_fallthrough", False):
            med_evidence_pro.MultiTierRetriever.retrieve = _wrap_retrieval_cache_fallthrough(current_retrieve)

        original_citation = med_evidence_pro.AnswerCascade._citation_complete
        if not getattr(original_citation, "_functionality_citation_guard", False):
            med_evidence_pro.AnswerCascade._citation_complete = staticmethod(_citation_complete_without_shared_state(original_citation))

        original_generate = med_evidence_pro.AnswerCascade.generate
        if not getattr(original_generate, "_functionality_generate_guard", False):
            med_evidence_pro.AnswerCascade.generate = _wrap_generate(original_generate)

        _INSTALLED = True


__all__ = [
    "install",
    "_wrap_retrieval_cache_fallthrough",
    "_wrap_retrieval_skip_health_probe",
    "_wrap_generate",
]
