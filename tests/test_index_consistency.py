from __future__ import annotations

import pytest

from rag_project.storage.vector_store import VectorStore
from rag_project.storage.vector_store import IndexCompatibilityError
from rag_project.embeddings import EmbeddingIdentity, EmbeddingProfile


def test_reconcile_index_removes_duplicate_chunk_metadata(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    documents = ["alpha text", "beta text"]
    metadatas = [
        {"document_id": "doc-42", "chunk_id": "chunk-7", "page_numbers": [1], "version_id": "v1", "index_state": "READY"},
        {"document_id": "doc-42", "chunk_id": "chunk-7", "page_numbers": [1], "version_id": "v1", "index_state": "READY"},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    ids = ["vec-1", "vec-2"]

    store.add_documents(documents, metadatas, embeddings, ids)
    result = store.reconcile_index("doc-42")

    assert result["valid"] is True
    final = store.validate_document_index("doc-42")
    assert final["count"] == 1
    assert final["valid"] is True


def test_lexical_index_is_persistent_and_returns_matching_chunks(tmp_path):
    path = tmp_path / "vectors"
    store = VectorStore(path)
    store.add_documents(
        ["treatment guidance for anemia", "unrelated cardiac anatomy"],
        [
            {"document_id": "doc-a", "chunk_id": "chunk-a", "page_numbers": [1], "version_id": "v1"},
            {"document_id": "doc-b", "chunk_id": "chunk-b", "page_numbers": [2], "version_id": "v1"},
        ],
        [[0.1, 0.2], [0.2, 0.3]],
        ["vec-a", "vec-b"],
    )

    reopened = VectorStore(path)
    result = reopened.search_lexical("anemia treatment", n_results=5)

    assert result["ids"][0] == ["vec-a"]
    assert result["metadatas"][0][0]["document_id"] == "doc-a"


def test_embedding_mismatch_is_reported_before_chroma_query(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    store.set_expected_identity(EmbeddingIdentity("test", "old", 2))
    store.add_documents(
        ["indexed"],
        [{"document_id": "doc", "chunk_id": "chunk", "page_numbers": [1], "version_id": "v1"}],
        [[0.1, 0.2]],
        ["vec"],
    )

    report = store.compatibility_report(EmbeddingIdentity("test", "new", 3))
    assert report["status"] == "INDEX_MIGRATION_REQUIRED"
    assert report["collection_dimension"] == 2
    with pytest.raises(IndexCompatibilityError, match="dimension mismatch"):
        store.search([0.1, 0.2, 0.3])


def test_embedding_profile_fingerprint_changes_for_different_models_and_dimensions():
    profile_old = EmbeddingProfile("ollama", model="qwen3-embedding:latest", dimension=384)
    profile_new_model = EmbeddingProfile("ollama", model="nomic-embed-text:latest", dimension=384)
    profile_new_dimension = EmbeddingProfile("ollama", model="qwen3-embedding:latest", dimension=4096)

    assert profile_old.fingerprint == profile_old.configuration_fingerprint
    assert profile_old.fingerprint != profile_new_model.fingerprint
    assert profile_old.fingerprint != profile_new_dimension.fingerprint


def test_profile_mismatch_without_dimension_change_still_requires_migration(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    profile_old = EmbeddingProfile("ollama", model="old-model", dimension=384)
    store.set_expected_identity(profile_old)
    store.add_documents(
        ["indexed"],
        [{"document_id": "doc", "chunk_id": "chunk", "page_numbers": [1], "version_id": "v1"}],
        [[0.1] * 384],
        ["vec"],
    )

    profile_new = EmbeddingProfile("ollama", model="new-model", dimension=384)
    report = store.compatibility_report(profile_new)
    assert report["status"] == "INDEX_MIGRATION_REQUIRED"
    assert "Index fingerprint does not match" in report["message"]


def test_empty_index_is_ready_for_new_embedding_profile(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    profile = EmbeddingProfile("ollama", model="qwen3-embedding:latest", dimension=4096)

    report = store.compatibility_report(profile)

    assert report["status"] == "READY"
    assert report["valid"] is True
    assert "no chunks have been indexed" in report["message"]
