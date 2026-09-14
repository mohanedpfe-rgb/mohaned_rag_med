from __future__ import annotations

from pathlib import Path

from rag_project.runtime_post_index_publication_contract import install as install_post_index_contract
from rag_project.storage.vector_store import VectorStore


def _store(tmp_path: Path) -> VectorStore:
    install_post_index_contract()
    return VectorStore(tmp_path / "vectors")


def _seed(store: VectorStore) -> None:
    store.add_documents(
        ["READY clinical chunk", "BUILDING stale clinical chunk", "FAILED superseded chunk"],
        [
            {"document_id": "ready-doc", "chunk_id": "ready-1", "version_id": "v1", "index_state": "READY"},
            {"document_id": "stale-doc", "chunk_id": "stale-1", "version_id": "v1", "index_state": "BUILDING"},
            {"document_id": "failed-doc", "chunk_id": "failed-1", "version_id": "v1", "index_state": "FAILED"},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0], [0.98, 0.02, 0.0, 0.0]],
        ["ready-1", "stale-1", "failed-1"],
    )


def test_semantic_search_returns_only_ready_chunks(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    result = store.search([1.0, 0.0, 0.0, 0.0], n_results=10)
    metadata = result["metadatas"][0]
    assert metadata
    assert all(str(item.get("index_state", "")).upper() == "READY" for item in metadata)
    assert {item.get("chunk_id") for item in metadata} == {"ready-1"}


def test_semantic_search_preserves_caller_filter_while_forcing_ready(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    result = store.search(
        [1.0, 0.0, 0.0, 0.0],
        n_results=10,
        where={"document_id": "stale-doc"},
    )
    assert result["ids"] == [[]]


def test_semantic_and_lexical_retrieval_have_same_publication_boundary(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    semantic = store.search([1.0, 0.0, 0.0, 0.0], n_results=10)
    lexical = store.search_lexical("clinical chunk", n_results=10)
    semantic_ids = {str(item) for item in semantic["ids"][0]}
    lexical_ids = {str(item) for item in lexical["ids"][0]}
    assert "stale-1" not in semantic_ids
    assert "failed-1" not in semantic_ids
    assert "stale-1" not in lexical_ids
    assert "failed-1" not in lexical_ids
