from __future__ import annotations

from pathlib import Path

from rag_project.runtime_post_index_publication_contract_v2 import install as install_post_index_v2
from rag_project.storage.vector_store import VectorStore


def test_version_delete_retries_semantic_cleanup_and_cleans_lexical_side(tmp_path: Path, monkeypatch):
    install_post_index_v2()
    store = VectorStore(tmp_path / "vectors")
    store.add_documents(
        ["version deletion regression"],
        [
            {
                "document_id": "delete-doc",
                "chunk_id": "delete-doc-v1-0",
                "version_id": "v1",
                "page_numbers": [1],
                "index_state": "BUILDING",
            }
        ],
        [[1.0, 0.0, 0.0, 0.0]],
        ["delete-doc-v1-0"],
    )
    calls = {"count": 0}
    original_collection_delete = store.collection.delete

    def fail_once(ids):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected semantic delete failure")
        return original_collection_delete(ids=ids)

    monkeypatch.setattr(store.collection, "delete", fail_once)

    store.delete_version("delete-doc", "v1")

    assert calls["count"] >= 2
    semantic = store.get_documents(where={"document_id": "delete-doc"})
    assert semantic.get("ids") in ([], None)
    assert store.lexical_count() == 0
