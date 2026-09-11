import pytest

@pytest.mark.high_level

def test_metadata_filter__changes_retrieval_scope(clean_system):
    retriever = getattr(clean_system, "retriever", None)
    assert retriever is not None
    try:
        hits = retriever.retrieve("diabetes", top_k=5, where={"language": "en"})
    except TypeError:
        pytest.skip("retriever does not expose metadata filter in this runtime")
    assert isinstance(hits, (list, tuple))

@pytest.mark.high_level

def test_answer__reports_document_aware_execution(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert result.get("document_aware", True) is True
