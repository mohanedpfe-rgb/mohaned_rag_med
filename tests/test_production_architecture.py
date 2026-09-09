from __future__ import annotations

from rag_project.app.production_rag import ProductionRAGSystem
from rag_project.application import runtime_contract
from rag_project.intelligence.final_44 import validate_feature_registry
from rag_project.intelligence.god_mode import GOD_MODE_FEATURES, report
from rag_project.intelligence.medical_safety import apply_medical_safety_policy, is_high_risk_medical_query


def test_production_pipeline_is_explicit() -> None:
    contract = runtime_contract()
    assert contract["service"] == "ProductionRAGSystem"
    assert contract["answer_pipeline"] == "explicit_delegation"
    assert contract["answer_monkey_patch"] is False
    assert contract["medical_safety_gate"] is True
    assert callable(ProductionRAGSystem.answer)


def test_44_feature_contract_is_executable_registry() -> None:
    assert len(GOD_MODE_FEATURES) == 44
    assert len(set(GOD_MODE_FEATURES)) == 44
    state = report()
    assert state["feature_count"] == 44
    assert state["answer_monkey_patch"] is False
    registry = validate_feature_registry()
    assert registry["count"] == 44
    assert registry["expected"] == 44
    assert registry["ok"] is True, registry["errors"]


def test_high_risk_medical_query_is_detected() -> None:
    assert is_high_risk_medical_query("What dose should I take for this treatment?")
    assert not is_high_risk_medical_query("What does this medical abbreviation mean?")


def test_high_risk_policy_abstains_without_verified_evidence() -> None:
    class Settings:
        medical_high_risk_evidence_threshold = 0.80

    result = apply_medical_safety_policy(
        "What medication should I take?",
        {"answer": "Take it.", "confidence": {"evidence_confidence": 0.50}, "citations": [], "grounding": {"allow": False}},
        Settings(),
    )
    assert result["status"] == "MEDICAL_SAFETY_ABSTAIN"
    assert result["citations"] == []
    assert "safely" in result["answer"].lower()
