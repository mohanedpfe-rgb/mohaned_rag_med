from rag_project.testing.production_retrieval_probes import phase9_independent_retrieval
from rag_project.testing.deep_diagnostics import PhaseSpec


def test_phase9_uses_independent_corpus_and_gold_labels():
    phase = PhaseSpec(9, "retrieval_microscope", "Retrieval microscope", "pytest", "")
    result = phase9_independent_retrieval(phase)
    assert result.status == "PASS", result.failures
    assert result.details["gold_labels_independent_of_corpus_text"] is True
    assert result.details["lexical_recall_at_3"] >= 0.80
    assert result.details["semantic_recall_at_3"] >= 0.80
    assert result.details["metadata_filter_correct"] is True
