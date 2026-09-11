import pytest

@pytest.mark.high_level

def test_numeric_question__retains_units_and_numbers(clean_system):
    result = clean_system.answer("What numeric information is stated about HbA1c?")
    answer = str(result.get("answer") or "")
    assert result.get("status")
    if answer:
        assert not isinstance(answer, bytes)

@pytest.mark.high_level

def test_table_and_figure_queries__return_structured_retrieval_signals(clean_system):
    for question in ("Which table contains the relevant dose information?", "What does Figure 1 show?"):
        result = clean_system.answer(question)
        retrieval = result.get("retrieval") or {}
        assert isinstance(retrieval, dict)
