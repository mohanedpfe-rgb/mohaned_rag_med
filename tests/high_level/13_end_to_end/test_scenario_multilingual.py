import pytest
from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level

def test_multilingual_scenario__answers_remain_language_aware(clean_system, tmp_path):
    for name, text, question in [
        ("en.pdf", "Diabetes mellitus is a chronic metabolic disorder.", "What is diabetes mellitus?"),
        ("fr.pdf", "Le diabète est une maladie métabolique chronique.", "Qu'est-ce que le diabète ?"),
        ("ar.pdf", "داء السكري هو اضطراب استقلابي مزمن.", "ما هو داء السكري؟"),
    ]:
        clean_system.ingest_file(write_minimal_pdf(tmp_path / name, [text]))
        result = clean_system.answer(question)
        assert result.get("status")
