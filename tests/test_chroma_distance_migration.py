from __future__ import annotations

import chromadb

from rag_project.runtime_chroma_distance_fix import install as install_distance_fix
from rag_project.storage.vector_store import VectorStore


def _seed_collection(path, *, count: int = 1) -> None:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "l2"},
    )
    if count:
        collection.add(
            ids=[f"chunk-{index}" for index in range(count)],
            documents=[f"document {index}" for index in range(count)],
            metadatas=[{"document_id": f"doc-{index}"} for index in range(count)],
            embeddings=[[1.0, 0.0, 0.0] for _ in range(count)],
        )


def test_distance_fix_migrates_existing_l2_collection_to_cosine(tmp_path):
    _seed_collection(tmp_path, count=1)
    install_distance_fix()

    store = VectorStore(tmp_path)

    assert store.collection.metadata.get("hnsw:space") == "cosine"
    assert store.count() == 1
    records = store.collection.get(include=["embeddings", "documents", "metadatas"])
    assert records["ids"] == ["chunk-0"]
    assert records["documents"] == ["document 0"]
    assert len(records["embeddings"]) == 1
    assert len(records["embeddings"][0]) == 3


def test_distance_fix_recreates_empty_mismatched_collection(tmp_path):
    _seed_collection(tmp_path, count=0)
    install_distance_fix()

    store = VectorStore(tmp_path)

    assert store.collection.metadata.get("hnsw:space") == "cosine"
    assert store.count() == 0
