from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any


def assert_status(result: dict[str, Any], allowed: set[str]) -> None:
    status = str(result.get("status") or "").upper()
    normalized = {value.upper() for value in allowed}
    assert status in normalized, f"status={status!r}, allowed={sorted(normalized)}"


def _citation_markers(result: dict[str, Any]) -> list[str]:
    answer = str(result.get("answer") or "")
    return re.findall(r"\[S\d+\]", answer, flags=re.I)


def assert_citations_valid(result: dict[str, Any]) -> None:
    markers = _citation_markers(result)
    citations = result.get("citations") or []
    hits = result.get("hits") or []
    status = str(result.get("status") or "").upper()
    if status in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE", "SYSTEM_NOT_READY", "ABSTAIN", "BLOCK"}:
        assert not markers, f"abstention must not claim source markers: {markers}"
        return
    assert markers or citations, "grounded success must expose citation markers or citation objects"
    if markers and hits:
        max_source = len(hits)
        for marker in markers:
            index_match = re.search(r"\d+", marker)
            assert index_match, f"malformed source marker {marker!r}"
            index = int(index_match.group())
            assert 1 <= index <= max_source, f"invalid source marker {marker} for {max_source} hits"


def assert_grounded(result: dict[str, Any]) -> None:
    assert_citations_valid(result)
    verification = result.get("verification") or result.get("grounding") or result.get("final_verification") or {}
    status = str(result.get("status") or "").upper()
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        ratio = verification.get("supported_ratio")
        if ratio is not None:
            assert float(ratio) >= 0.60, f"grounding ratio too low: {ratio}"
        answer = str(result.get("answer") or "").strip()
        assert answer, "successful answer must not be empty"
        assert verification.get("allow") is not False, verification


def assert_no_llm_called(llm_spy: Any) -> None:
    assert not getattr(llm_spy, "calls", []), f"unexpected LLM calls: {len(llm_spy.calls)}"


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


def assert_latency_under(result: dict[str, Any], seconds: float) -> None:
    latency = result.get("latency_ms")
    if latency is None:
        trace = result.get("query_trace") or {}
        latency = (trace.get("timings_ms") or {}).get("total")
    assert latency is not None, "result must expose measurable latency"
    assert float(latency) <= seconds * 1000.0, f"latency {latency}ms > {seconds}s"


def assert_extractive_path(result: dict[str, Any]) -> None:
    path = str(result.get("generation_path") or (result.get("answer_plan") or {}).get("selected_path") or "").upper()
    assert path in {"PATH_A_EXTRACTIVE", "PATH_A_VERIFIED_FALLBACK"}, f"unexpected extractive path: {path!r}"


def assert_abstained(result: dict[str, Any]) -> None:
    assert_status(result, {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE", "SYSTEM_NOT_READY", "ABSTAIN", "BLOCK"})
    answer = str(result.get("answer") or "").lower()
    assert not any(token in answer for token in ("i am certain", "definitely", "the patient should"))
    assert not result.get("citations"), "abstention must not expose positive citations"


def assert_numeric_preserved(result: dict[str, Any], expected_numbers: Iterable[str]) -> None:
    answer = str(result.get("answer") or "")
    for expected in expected_numbers:
        assert str(expected) in answer, f"missing exact numeric token {expected!r}"


def assert_entities_present(result: dict[str, Any], entities: Iterable[str]) -> None:
    analysis = result.get("query_analysis") or result.get("route") or {}
    present = " ".join(str(x) for x in analysis.get("entities", [])).casefold()
    for entity in entities:
        assert entity.casefold() in present or entity.casefold() in str(result).casefold()


def assert_document_ready(system: Any, document_id: str) -> None:
    assert document_id
    candidates = [getattr(system, "ingestion_store", None), getattr(system, "registry", None), getattr(system, "document_registry", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for method_name in ("get", "get_document", "status"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    value = method(document_id)
                    if isinstance(value, dict):
                        assert str(value.get("status", "")).upper() == "READY"
                        return
                    if str(value).upper() == "READY":
                        return
                except Exception:
                    pass
    raise AssertionError(f"unable to verify READY state for document {document_id}")


def assert_no_history_contamination(system: Any, question: str) -> None:
    memory = getattr(system, "conversation_memory", None)
    if memory is None:
        return
    rendered = str(memory).casefold()
    assert question.casefold() not in rendered or rendered.count(question.casefold()) <= 1


def timed_call(callable_, *args, **kwargs):
    started = time.perf_counter()
    result = callable_(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return result, elapsed
