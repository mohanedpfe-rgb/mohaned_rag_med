from __future__ import annotations

import math
from pathlib import Path

from rag_project.intelligence.semantic_cache import SemanticRetrievalCache
from rag_project.retrieval.hybrid_retriever import RetrievalHit


def _embedder(mapping: dict[str, list[float]]):
    return lambda text: mapping[text]


def _hit(text: str) -> RetrievalHit:
    return RetrievalHit("doc-1", text, {"document_id": "doc-1", "page": 1}, 0.9, 0.8, 0.7)


def test_semantic_cache_matches_similar_queries_above_threshold(tmp_path: Path):
    first = [1.0] + [0.0] * 767
    similar = [0.99999] + [0.0] * 766 + [0.004]
    mapping = {"original": first, "paraphrase": similar}
    cache = SemanticRetrievalCache(tmp_path / "cache.sqlite3", embed_query=_embedder(mapping), similarity_threshold=0.95)
    assert cache.put("original", [_hit("evidence")])
    result = cache.get("paraphrase")
    assert result is not None
    hits, metadata = result
    assert hits[0].text == "evidence"
    assert metadata["similarity"] >= 0.95


def test_semantic_cache_rejects_below_threshold(tmp_path: Path):
    mapping = {"a": [1.0] + [0.0] * 767, "b": [0.0, 1.0] + [0.0] * 766}
    cache = SemanticRetrievalCache(tmp_path / "cache.sqlite3", embed_query=_embedder(mapping), similarity_threshold=0.95)
    cache.put("a", [_hit("evidence")])
    assert cache.get("b") is None


def test_semantic_cache_enforces_ttl_and_capacity(tmp_path: Path):
    mapping = {f"q{i}": [float(i + 1)] + [0.0] * 767 for i in range(3)}
    cache = SemanticRetrievalCache(tmp_path / "cache.sqlite3", embed_query=_embedder(mapping), ttl_seconds=0, max_entries=2)
    cache.put("q0", [_hit("a")]); cache.put("q1", [_hit("b")]); cache.put("q2", [_hit("c")])
    assert cache.stats()["entries"] <= 2
    assert cache.get("q0") is None


def test_semantic_cache_contract_defaults_match_plan(tmp_path: Path):
    cache = SemanticRetrievalCache(tmp_path / "cache.sqlite3", embed_query=lambda _: [1.0] + [0.0] * 767)
    stats = cache.stats()
    assert math.isclose(stats["similarity_threshold"], 0.95)
    assert stats["ttl_seconds"] == 7 * 24 * 60 * 60
    assert stats["max_entries"] == 10_000
    assert stats["expected_dimension"] == 768
    assert stats["semantic"] is True
