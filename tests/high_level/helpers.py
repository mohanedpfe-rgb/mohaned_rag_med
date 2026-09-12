from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any


ABSTAIN_STATUSES = {
    "NOT_SUPPORTED",
    "REASONING_ABSTAIN",
    "ANSWER_UNAVAILABLE",
    "SYSTEM_NOT_READY",
    "ABSTAIN",
    "BLOCK",
    "GENERATION_ABSTAIN",
}
SUCCESS_STATUSES = {"SUCCESS", "SUCCESS_WITH_WARNINGS"}


def assert_status(result: dict[str, Any], allowed: set[str]) -> None:
    status = str(result.get("status") or "").upper()
    normalized = {value.upper() for value in allowed}
    assert status in normalized, f"status={status!r}, allowed={sorted(normalized)}"


def assert_exact_status(result: dict[str, Any], expected: str) -> None:
    actual = str(result.get("status") or "").upper()
    expected_upper = expected.upper()
    assert actual == expected_upper, f"expected exact status {expected_upper!r}, got {actual!r}"


def assert_exact_path(result: dict[str, Any], expected: str) -> None:
    actual = str(result.get("generation_path") or "").upper()
    expected_upper = expected.upper()
    assert actual == expected_upper, f"expected exact generation path {expected_upper!r}, got {actual!r}"
    plan_path = str((result.get("answer_plan") or {}).get("selected_path") or "").upper()
    if plan_path:
        assert plan_path == expected_upper, f"answer plan path {plan_path!r} diverges from generation path {actual!r}"


def assert_pipeline_authority(result: dict[str, Any]) -> None:
    authority = str(result.get("pipeline_authority") or result.get("implementation_authority") or "")
    assert authority == "rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine", authority
    assert result.get("canonical_pipeline_executed") is True


def _citation_markers(result: dict[str, Any]) -> list[str]:
    answer = str(result.get("answer") or "")
    return re.findall(r"\[S\d+\]", answer, flags=re.I)


def assert_citations_valid(result: dict[str, Any]) -> None:
    markers = _citation_markers(result)
    citations = list(result.get("citations") or [])
    hits = list(result.get("hits") or [])
    status = str(result.get("status") or "").upper()

    if status in ABSTAIN_STATUSES:
        assert not markers, f"abstention must not claim source markers: {markers}"
        assert not citations, "abstention must not expose positive citations"
        return

    assert status in SUCCESS_STATUSES, f"citation validation requires a terminal success state, got {status!r}"
    assert hits, "successful grounded answer must retain its retrieved evidence"
    assert markers or citations, "grounded success must expose citation markers or citation objects"

    if markers:
        max_source = len(hits)
        for marker in markers:
            index_match = re.fullmatch(r"\[S(\d+)\]", marker, flags=re.I)
            assert index_match, f"malformed source marker {marker!r}"
            index = int(index_match.group(1))
            assert 1 <= index <= max_source, f"invalid source marker {marker} for {max_source} hits"

    if citations:
        valid = [citation for citation in citations if isinstance(citation, dict) and citation.get("valid") is True]
        assert len(valid) == len(citations), f"all returned citations must be validated: {citations!r}"
        hit_keys = {
            (
                str((hit.metadata or {}).get("document_id", hit.doc_id)),
                str((hit.metadata or {}).get("chunk_id", "")),
                tuple((hit.metadata or {}).get("page_numbers", (hit.metadata or {}).get("source_pages", [])) or []),
            )
            for hit in hits
        }
        for citation in citations:
            pages = tuple(citation.get("page_numbers", []) or [])
            key = (
                str(citation.get("document_id", "")),
                str(citation.get("chunk_id", "")),
                pages,
            )
            assert any(
                key[0] == hit_key[0]
                and (not key[1] or key[1] == hit_key[1])
                and (not key[2] or bool(set(key[2]) & set(hit_key[2])))
                for hit_key in hit_keys
            ), f"citation does not resolve to a retrieved hit: {citation!r}"


def assert_grounded(result: dict[str, Any]) -> None:
    assert_citations_valid(result)
    status = str(result.get("status") or "").upper()
    if status in SUCCESS_STATUSES:
        verification = result.get("verification") or result.get("grounding") or result.get("final_verification") or {}
        ratio = verification.get("supported_ratio")
        assert ratio is not None, "successful answer must expose a measured supported_ratio"
        assert float(ratio) >= 0.70, f"grounding ratio too low: {ratio}"
        assert verification.get("allow") is True, verification
        assert str(result.get("generation_path") or ""), "successful answer must expose exact generation path"
        assert_pipeline_authority(result)
        assert str(result.get("answer") or "").strip(), "successful answer must not be empty"


def assert_no_llm_called(llm_spy: Any) -> None:
    calls = getattr(llm_spy, "calls", [])
    assert not calls, f"unexpected LLM calls: {len(calls)}"


def assert_llm_called_with_small_context(llm_spy: Any, max_tokens: int = 700) -> None:
    calls = getattr(llm_spy, "calls", [])
    assert calls, "expected at least one LLM call"
    for call in calls:
        prompt = str(call.get("prompt", ""))
        approx_tokens = max(1, len(prompt.split()))
        assert approx_tokens <= max_tokens, f"LLM context too large: ~{approx_tokens} tokens"
        kwargs = call.get("kwargs") or {}
        if "temperature" in kwargs:
            assert float(kwargs["temperature"]) == 0.0


def assert_prompt_excludes(llm_spy: Any, forbidden: Iterable[str]) -> None:
    prompts = "\n".join(str(call.get("prompt") or "") for call in getattr(llm_spy, "calls", []))
    for token in forbidden:
        assert str(token) not in prompts, f"forbidden content leaked into LLM prompt: {token!r}"


def assert_latency_under(result: dict[str, Any], seconds: float) -> None:
    latency = result.get("latency_ms")
    if latency is None:
        trace = result.get("query_trace") or {}
        latency = (trace.get("timings_ms") or {}).get("total")
    assert latency is not None, "result must expose measurable latency"
    assert float(latency) <= seconds * 1000.0, f"latency {latency}ms > {seconds}s"


def assert_extractive_path(result: dict[str, Any]) -> None:
    assert_exact_path(result, "PATH_A_EXTRACTIVE")


def assert_abstained(result: dict[str, Any]) -> None:
    assert_status(result, ABSTAIN_STATUSES)
    answer = str(result.get("answer") or "").lower()
    assert not any(token in answer for token in ("i am certain", "definitely", "the patient should"))
    assert not result.get("citations"), "abstention must not expose positive citations"


def assert_numeric_preserved(result: dict[str, Any], expected_numbers: Iterable[str]) -> None:
    answer = str(result.get("answer") or "")
    for expected in expected_numbers:
        assert str(expected) in answer, f"missing exact numeric token {expected!r}"


def assert_rank_contains(hits: Iterable[Any], token: str, top_k: int = 3) -> None:
    ordered = list(hits)[:top_k]
    needle = str(token).casefold()
    assert any(needle in str(getattr(hit, "text", "")).casefold() for hit in ordered), (
        f"expected {token!r} within top {top_k} hits; got {[getattr(hit, 'text', '') for hit in ordered]!r}"
    )


def assert_entities_present(result: dict[str, Any], entities: Iterable[str]) -> None:
    route = result.get("route") or {}
    analysis = result.get("query_analysis") or {}
    route_entities = route.get("entities", []) if isinstance(route, dict) else []
    analysis_entities = analysis.get("entities", []) if isinstance(analysis, dict) else []
    present = " ".join(str(x) for x in [*route_entities, *analysis_entities]).casefold()
    for entity in entities:
        assert entity.casefold() in present, f"expected entity {entity!r} was not extracted: {route_entities or analysis_entities!r}"


def assert_document_ready(system: Any, document_id: str) -> dict[str, Any]:
    assert document_id, "a ready document must have a document_id"
    state_store = getattr(system, "state_store", None)
    assert state_store is not None, "production system must expose its authoritative ingestion state store"
    getter = getattr(state_store, "get_document", None)
    assert callable(getter), "ingestion state store must expose get_document"
    document = getter(document_id)
    assert isinstance(document, dict), f"document {document_id!r} is missing from durable state"
    status = str(document.get("status") or "").upper()
    index_state = str(document.get("index_state") or "").upper()
    assert status == "READY", f"document status is {status!r}, expected READY"
    assert index_state == "READY", f"document index_state is {index_state!r}, expected READY"
    assert int(document.get("total_pages") or 0) > 0, document
    assert int(document.get("current_page") or 0) == int(document.get("total_pages") or 0), document
    return document


def assert_no_history_contamination(system: Any, question: str) -> None:
    memory = getattr(system, "conversation_memory", None)
    assert memory is not None, "production system must expose conversation memory"
    history = list(getattr(memory, "history", []) or [])
    occurrences = sum(1 for item in history if question.casefold() in str(item).casefold())
    assert occurrences <= 1, f"question appears too many times in conversation memory: {occurrences}"


def assert_memory_unchanged(before: Iterable[Any], after: Iterable[Any]) -> None:
    assert list(before) == list(after), "conversation memory changed during a non-storable answer"


def timed_call(callable_, *args, **kwargs):
    started = time.perf_counter()
    result = callable_(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return result, elapsed
