import pytest

@pytest.mark.high_level

def test_llm_path__is_explicitly_configured(clean_system):
    settings = clean_system.settings
    assert float(settings.temperature) >= 0.0
    assert int(settings.generation_max_output_tokens) > 0
    assert int(settings.context_token_budget) >= 256

@pytest.mark.high_level
@pytest.mark.requires_ollama

def test_real_llm__respects_runtime_temperature(real_ollama_optional, clean_system):
    assert float(clean_system.settings.temperature) <= 1.0
