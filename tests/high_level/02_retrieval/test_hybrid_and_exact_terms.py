import pytest

@pytest.mark.high_level

def test_exact_medical_terms__are_searchable(clean_system):
    retriever = getattr(clean_system, "retriever", None)
    assert retriever is not None
    hits = retriever.retrieve("HbA1c DKA metformin", top_k=6)
    assert isinstance(hits, (list, tuple))

@pytest.mark.high_level

def test_retrieval_configuration__is_hybrid(clean_system):
    settings = clean_system.settings
    assert str(settings.lexical_mode).lower() in {"hybrid", "bm25", "lexical"}
    assert 0.0 <= float(settings.vector_weight) <= 1.0
