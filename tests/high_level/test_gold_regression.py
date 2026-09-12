from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from tests.high_level.helpers import (
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_latency_under,
    assert_no_llm_called,
    assert_pipeline_authority,
)


def pytest_generate_tests(metafunc):
    if "gold_case" not in metafunc.fixturenames:
        return
    root = Path(__file__).resolve().parents[1] / "support" / "gold_sets" / "core.jsonl"
    cases = [json.loads(line) for line in root.read_text(encoding="utf-8").splitlines() if line.strip()]
    metafunc.parametrize("gold_case", cases, ids=[str(case["id"]) for case in cases])


def _grounded_spy_response(system, question: str, expected_terms: list[str]) -> str:
    hits = list(system.retriever.retrieve(question, top_k=3, where=None) or [])
    parts: list[str] = []
    for index, hit in enumerate(hits[:3], start=1):
        text = " ".join(str(getattr(hit, "text", "") or "").split())
        if not text:
            continue
        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        if len(sentence) < 18:
            sentence = text[:500].strip()
        parts.append(f"{sentence} [S{index}]")
    if parts:
        return "\n".join(parts)
    return ". ".join(expected_terms or ["No evidence was retrieved"]) + ". [S1]"


@pytest.mark.high_level
def test_gold_case__matches_exact_status_path_content_language_latency_and_memory_contract(clean_system, fake_ollama_fast, gold_case):
    case = gold_case
    expected_status = str(case["expected_status"]).upper()
    expected_path = str(case["expected_path"]).upper()
    expected_language = str(case["language"]).lower()
    must_contain = [str(value).casefold() for value in case.get("must_contain", [])]
    must_not_contain = [str(value).casefold() for value in case.get("must_not_contain", [])]
    memory = clean_system.conversation_memory
    before_history = list(getattr(memory, "history", []) or [])

    if expected_path == "PATH_C_CONSTRAINED_LLM":
        fake_ollama_fast.response = _grounded_spy_response(clean_system, case["question"], [str(v) for v in case.get("must_contain", [])])
    else:
        terms = [str(value) for value in case.get("must_contain", [])]
        fake_ollama_fast.response = "The indexed evidence supports these facts: " + ", ".join(terms or ["the indexed medical evidence"]) + ". [S1]"
    clean_system.llm = fake_ollama_fast

    started = time.perf_counter()
    result = clean_system.answer(case["question"])
    wall_clock = time.perf_counter() - started

    assert_exact_status(result, expected_status)
    if expected_path == "NO_GENERATION_PATH":
        assert not result.get("generation_path"), result
    else:
        assert_exact_path(result, expected_path)

    answer = str(result.get("answer") or "")
    answer_folded = answer.casefold()
    for token in must_contain:
        assert token in answer_folded, f"gold case {case['id']} missing required content {token!r}: {answer!r}"
    for token in must_not_contain:
        assert token not in answer_folded, f"gold case {case['id']} leaked forbidden content {token!r}: {answer!r}"

    max_latency = float(case["max_latency_s"])
    assert wall_clock <= max_latency, f"gold case {case['id']} wall-clock latency {wall_clock:.3f}s exceeded {max_latency:.3f}s"

    route = result.get("route") or {}
    assert str(route.get("language") or "").lower() == expected_language, (
        f"gold case {case['id']} expected language {expected_language!r}, got {route!r}"
    )

    after_history = list(getattr(memory, "history", []) or [])
    if expected_status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert len(after_history) == len(before_history) + 1
        assert case["question"] in str(after_history[-1])
        assert_latency_under(result, max_latency)
        assert_citations_valid(result)
        assert_grounded(result)
        assert_pipeline_authority(result)
        if case.get("must_cite"):
            assert result.get("citations") or "[S" in answer
    else:
        assert after_history == before_history
        assert result.get("citations") == []

    if case.get("must_not_call_llm"):
        assert_no_llm_called(fake_ollama_fast)
    else:
        assert fake_ollama_fast.calls, f"gold case {case['id']} expected an LLM call"
