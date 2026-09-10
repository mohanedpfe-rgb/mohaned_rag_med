from __future__ import annotations

import json

from rag_project.intelligence.query_intelligence import plan_query
from rag_project.intelligence.semantic_reasoning import understand_query
from rag_project.intelligence.small_model_reasoner import (
    analyze_with_small_model,
    augment_query_plan,
    merge_understanding,
    should_use_small_model,
)


class FakeLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return json.dumps(self.payload)


def test_easy_definition_does_not_escalate():
    understanding = understand_query("What is diabetes?")
    assert should_use_small_model("What is diabetes?", understanding) is False


def test_hard_followup_escalates():
    understanding = understand_query("And the treatment?", conversation_context="User asked about diabetic ketoacidosis diagnosis.")
    assert should_use_small_model("And the treatment?", understanding) is True


def test_small_model_output_is_strictly_bounded_and_parsed():
    llm = FakeLLM({
        "intent": "management",
        "entities": ["diabetic ketoacidosis"],
        "relations": ["association", "invalid_relation"],
        "constraints": ["population", "invalid"],
        "subquestions": ["DKA treatment in children"],
        "retrieval_terms": ["initial management", "contraindications"],
        "answer_strategy": "Compare only evidence-backed treatment pathways.",
    })
    understanding = understand_query("What treatment should be used for DKA in children?")
    result = analyze_with_small_model(llm, "What treatment should be used for DKA in children?", understanding)
    assert llm.calls == 1
    assert result is not None
    assert result["intent"] == "management"
    assert result["relations"] == ["association"]
    assert result["constraints"] == ["population"]
    assert len(result["retrieval_terms"]) <= 6


def test_assist_can_expand_retrieval_without_replacing_deterministic_core():
    understanding = understand_query("Why does DKA cause hypokalemia?")
    plan = plan_query("Why does DKA cause hypokalemia?")
    augmented = augment_query_plan(plan, {
        "intent": "etiology",
        "entities": ["diabetic ketoacidosis", "hypokalemia", "insulin shift"],
        "subquestions": ["DKA mechanism of hypokalemia"],
        "retrieval_terms": ["potassium shift"],
        "relations": ["causality"],
    })
    assert "insulin shift" in augmented.entities
    assert any("potassium shift" in item for item in augmented.variants)
    assert augmented.needs_multi_hop is True
    assert "hypokalemia" in augmented.entities
    merged = merge_understanding(understanding, {"intent": "etiology", "relations": ["causality"], "constraints": []})
    assert merged.primary_intent == "etiology"
    assert "causality" in merged.relations


def test_invalid_or_unavailable_small_model_falls_back_cleanly():
    class BrokenLLM:
        def generate(self, **kwargs):
            raise RuntimeError("offline")

    understanding = understand_query("Why does diabetes cause albuminuria?")
    assert analyze_with_small_model(BrokenLLM(), "Why does diabetes cause albuminuria?", understanding) is None
