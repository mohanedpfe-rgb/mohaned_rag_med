from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from rag_project.generation.latency_budget import configured_budget, exhausted, remaining, request_budget
from rag_project.intelligence.god_mode import _simple_extractive_answer
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.app.production_rag import _is_explicit_followup
from rag_project.app.rag_system import RetrievalHit


def _hit(text: str, score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        doc_id="doc-1",
        text=text,
        metadata={"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
        score=score,
        vector_score=score,
        lexical_score=0.0,
    )


def test_request_budget_has_predictable_hard_cap():
    settings = SimpleNamespace(generation_latency_budget_seconds=120.0)
    assert configured_budget(settings) == 45.0


def test_request_budget_preserves_configured_budget_below_cap():
    settings = SimpleNamespace(generation_latency_budget_seconds=12.5)
    assert configured_budget(settings) == 12.5


def test_request_budget_expires_and_reports_remaining_time():
    settings = SimpleNamespace(generation_latency_budget_seconds=1.0)
    with request_budget(settings):
        assert not exhausted()
        assert 0.0 < (remaining() or 0.0) <= 1.0
        time.sleep(1.05)
        assert exhausted()
        assert remaining() == 0.0


def test_simple_extractive_answer_never_calls_generation_and_cleans_metadata():
    answer = _simple_extractive_answer(
        "What is phosphate regulation?",
        [_hit("[Section: 3. Régulation :] Le rein est le principal site de régulation de la concentration plasmatique du Pi.")],
    )
    assert "Le rein est le principal site" in answer
    assert "[Section: 3. Régulation :]" not in answer


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What is HbA1c?",
        "What is the main site of phosphate regulation?",
        "What are the complications?",
        "Define metabolic acidosis.",
    ],
)
def test_common_factual_questions_do_not_require_table_retrieval(question):
    plan = plan_query(question)
    assert plan.needs_table is False


@pytest.mark.parametrize(
    "question",
    [
        "Which table contains potassium values?",
        "Show the table of electrolyte concentrations.",
        "What are the rows and columns in this table?",
    ],
)
def test_explicit_table_queries_still_require_table_retrieval(question):
    plan = plan_query(question)
    assert plan.needs_table is True


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is diabetes?", False),
        ("What is HbA1c?", False),
        ("What about complications?", True),
        ("And treatment?", True),
        ("والمضاعفات؟", True),
        ("Then what happens?", True),
    ],
)
def test_followup_detection_requires_explicit_followup_signal(question, expected):
    assert _is_explicit_followup(question) is expected


def test_short_standalone_question_is_not_followup_even_after_history_exists():
    # This regression targets the old broad <=10-token heuristic. The production
    # route must preserve short standalone questions without rewriting them.
    assert _is_explicit_followup("What is diabetes?") is False
