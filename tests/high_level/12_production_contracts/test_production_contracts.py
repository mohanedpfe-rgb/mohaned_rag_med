import pytest

@pytest.mark.high_level

def test_single_answer_authority__is_canonical(clean_system):
    from rag_project.application import ACTIVE_ANSWER_PIPELINE_AUTHORITY
    result = clean_system.answer("What is diabetes mellitus?")
    assert result.get("pipeline_authority") == ACTIVE_ANSWER_PIPELINE_AUTHORITY

@pytest.mark.high_level

def test_runtime_contract__points_to_composition_root():
    from rag_project.application import runtime_contract
    contract = runtime_contract()
    assert contract["composition_root"] == "rag_project.application.create_rag_system"
    assert contract["answer_pipeline"] == "med_evidence_pro"
    assert contract["answer_monkey_patch"] is False

@pytest.mark.high_level

def test_startup_quality__is_explicit(clean_system):
    quality = getattr(clean_system, "startup_quality", None)
    assert isinstance(quality, dict)
    assert "ready" in quality
