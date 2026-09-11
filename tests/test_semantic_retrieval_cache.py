from __future__ import annotations

import math
from pathlib import Path

from rag_project.intelligence.semantic_cache import SemanticRetrievalCache
from rag_project.retrieval.hybrid_retriever import RetrievalHit


def test_semantic_cache_matches_near_duplicate_query_and_preserves_hits(tmp_path: Path):
    vectors = {
        "q1": [1.0, 0.0, 0.0],
        "q1-near": [0.9999, 0.014, 0.0],
        "q2": [0.0, 1.0, 0.0],
    }
    cache = SemanticRetrievalCache(
        tmp_path / "cache.sqlite3",
        embed_query=lambda q: vectors[q],
        similarity_threshold=0.95,
        max_entries=10_000,
        expected_dimension=3,
    )
    hit = RetrievalHit("doc-1", "Evidence text", {"page": 1}, 0.9, 0.8, 0.7)

    assert cache.put("q1", [hit])
    result = cache.get("q1-near")
    assert result is not None
    restored, metadata = result
    assert restored[0].doc_id == "doc-1"
    assert metadata["similarity"] >= 0.95
    assert metadata["hits"] == 1


def test_semantic_cache_rejects_low_similarity_and_expires(tmp_path: Path):
    now = {"value": 1000.0}
    import rag_project.intelligence.semantic_cache as module

    original_time = module.time.time
    module.time.time = lambda: now["value"]
    try:
        cache = SemanticRetrievalCache(
            tmp_path / "cache.sqlite3",
            embed_query=lambda q: {"a": [1.0, 0.0], "b": [0.0, 1.0]}[q],
            similarity_threshold=0.95,
            ttl_seconds=10,
            expected_dimension=2,
        )
        hit = RetrievalHit("doc", "text", {}, 1.0, 1.0, 1.0)
        assert cache.put("a", [hit])
        assert cache.get("b") is None
        now["value"] = 1011.0
        assert cache.get("a") is None
    finally:
        module.time.time = original_time


def test_semantic_cache_cosine_is_finite_and_exact_for_identical_vectors():
    value = SemanticRetrievalCache.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert math.isfinite(value)
    assert value == 1.0
