from __future__ import annotations

import json
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


@pytest.mark.high_level
def test_gold_case__matches_exact_status_path_content_and_latency(clean_system, fake_ollama_fast, gold_case):
    case = gold_case
    expected_status = str(case["expected_status"]).upper()
    expected_path = str(case["expected_path"]).upper()
    must_contain = [str(value).casefold() for value in case.get("must_contain", [])]
    must_not_contain = [str(value).casefold() for value in case.get("must_not_contain", [])]

    terms = [str(value) for value in case.get("must_contain", [])]
    fake_ollama_fast.response = (
        "The indexed evidence supports these facts: "
        + ", ".join(terms or ["the indexed medical evidence"])
        + ". [S1]"
    )
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

    if expected_status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_latency_under(result, max_latency)
        assert_citations_valid(result)
        assert_grounded(result)
        assert_pipeline_authority(result)
        if case.get("must_cite"):
            assert result.get("citations") or "[S" in answer
    else:
        assert result.get("citations") == []

    if case.get("must_not_call_llm"):
        assert_no_llm_called(fake_ollama_fast)
    else:
        assert fake_ollama_fast.calls, f"gold case {case['id']} expected an LLM call"
