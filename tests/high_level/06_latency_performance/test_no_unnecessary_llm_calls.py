import pytest

@pytest.mark.high_level

def test_simple_factual_answer__exposes_non_llm_fast_path_signal(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    path = str(result.get("generation_path") or (result.get("answer_plan") or {}).get("selected_path") or "").casefold()
    assert path
    assert "llm" not in path or "extract" in path

@pytest.mark.high_level

def test_runtime__has_bounded_ollama_concurrency(clean_system):
    assert 1 <= int(clean_system.settings.ollama_concurrency) <= 2
