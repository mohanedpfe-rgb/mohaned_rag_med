from __future__ import annotations

from rag_project.reranking.reranker import Reranker
from rag_project.retrieval.hybrid_retriever import RetrievalHit


def _hit(chunk_id: str, score: float = 0.8) -> RetrievalHit:
    return RetrievalHit(
        doc_id="doc-1",
        text=f"Evidence for {chunk_id}",
        metadata={"document_id": "doc-1", "chunk_id": chunk_id},
        score=score,
    )


def test_cross_encoder_logits_are_normalized_to_probability_range():
    reranker = Reranker(enabled=True)
    reranker._model_attempted = True

    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [-9.0, 0.0, 9.0]

    reranker.model = FakeModel()
    hits = reranker.rerank("dose", [_hit("low"), _hit("mid"), _hit("high")])

    assert [hit.metadata["chunk_id"] for hit in hits] == ["high", "mid", "low"]
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
    assert hits[0].score > 0.99
    assert 0.49 < hits[1].score < 0.51
    assert hits[2].score < 0.01


def test_non_finite_reranker_outputs_fail_safe_to_zero():
    reranker = Reranker(enabled=True)
    reranker._model_attempted = True

    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [float("nan")]

    reranker.model = FakeModel()
    hits = reranker.rerank("dose", [_hit("nan")])

    assert len(hits) == 1
    assert hits[0].score == 0.0
