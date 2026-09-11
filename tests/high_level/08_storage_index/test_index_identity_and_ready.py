import pytest

@pytest.mark.high_level

def test_embedding_identity__is_part_of_settings(clean_system):
    assert str(clean_system.settings.embedding_model).strip()

@pytest.mark.high_level

def test_runtime__exposes_vector_store(clean_system):
    candidates = [getattr(clean_system, "vector_store", None), getattr(clean_system, "retriever", None)]
    assert any(candidate is not None for candidate in candidates)
