import re
import pytest

@pytest.mark.high_level

def test_numeric_question__preserves_source_numbers_and_units(clean_system):
    result = clean_system.answer("What numeric information is stated about HbA1c?")
    answer = str(result.get("answer") or "")
    assert not re.search(r"\b\d+(?:\.\d+)?\s*%", answer) or "HbA1c" in answer

@pytest.mark.high_level

def test_answer_plan__declares_selected_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    plan = result.get("answer_plan") or {}
    assert "selected_path" in plan or "generation_path" in result
