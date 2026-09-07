from __future__ import annotations

from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.retrieval.metadata_filter import MetadataFilter
from rag_project.reranking.reranker import Reranker


def _hit(index: int, text: str = "medical evidence") -> RetrievalHit:
    return RetrievalHit(
        doc_id="doc-1",
        text=text,
        metadata={
            "document_id": "doc-1",
            "chunk_id": f"chunk-{index}",
            "chunk_index": index,
            "file_name": "test.pdf",
            "page_numbers": [index + 1],
            "index_state": "READY",
        },
        score=1.0 - index / 10,
        vector_score=0.8 - index / 10,
        lexical_score=0.7 - index / 10,
    )


def test_reranker_does_not_load_model_until_needed():
    reranker = Reranker()
    assert reranker.model is None
    assert reranker.rerank("question", [_hit(0)])[0].doc_id == "doc-1"


def test_context_builder_can_fetch_missing_neighbors():
    def resolver(document_id: str, index: int, radius: int):
        return [_hit(index - 1)] if index > 0 else []

    builder = ContextBuilder(neighbor_expansion=True, neighbor_resolver=resolver)
    context, selected = builder.build([_hit(2)])
    assert "chunk-2" in context
    assert any(hit.metadata["chunk_id"] == "chunk-1" for hit in selected)


def test_metadata_filter_does_not_create_empty_and_clause():
    assert MetadataFilter.build({"language": None, "file_name": ""}) == {"index_state": "READY"}


def test_confidence_is_empty_for_no_hits():
    assert Reranker.confidence([]) == {"level": "none", "top_score": 0.0, "margin": 0.0}
