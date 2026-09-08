from __future__ import annotations

import json
import sqlite3

from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.hybrid_retriever import RetrievalHit
from rag_project.retrieval.metadata_filter import MetadataFilter
from rag_project.reranking.reranker import Reranker
from rag_project.runtime_hardening import _safe_lexical_search
from rag_project.runtime_hardening_extra import _safe_ingestion_version_id


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


class _FakeVectorStore:
    def __init__(self, db_path):
        self.lexical_database = db_path

    @staticmethod
    def _as_query_result(ids, documents, metadatas, distances=None):
        return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [distances or []]}

    @staticmethod
    def _lexical_tokens(text):
        return text.casefold().split()

    @staticmethod
    def _coerce_metadata(metadata):
        return metadata

    @staticmethod
    def _metadata_matches(meta, where):
        return not where or all(meta.get(k) == v for k, v in where.items())


def test_lexical_search_counts_document_frequency_by_query_token(tmp_path):
    db = tmp_path / "lexical.sqlite3"
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE lexical_documents (id TEXT PRIMARY KEY, document TEXT, metadata TEXT, index_state TEXT, tokens TEXT)"
        )
        con.executemany(
            "INSERT INTO lexical_documents VALUES (?, ?, ?, ?, ?)",
            [
                ("1", "alpha beta", json.dumps({"index_state": "READY"}), "READY", json.dumps(["alpha", "beta"])),
                ("2", "alpha", json.dumps({"index_state": "READY"}), "READY", json.dumps(["alpha"])),
            ],
        )
    result = _safe_lexical_search(_FakeVectorStore(db), "beta", n_results=5)
    assert result["ids"][0] == ["1"]


def test_ingestion_version_changes_when_embedding_profile_changes():
    base = {
        "content_hash": "abc",
        "parser_version": "pdf-extractor-v2",
        "ocr_config": {"enabled": False},
        "chunking_config": {"size": 600, "overlap": 100},
        "embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
    }
    assert _safe_ingestion_version_id(**base, embedding_profile="profile-a") != _safe_ingestion_version_id(**base, embedding_profile="profile-b")
