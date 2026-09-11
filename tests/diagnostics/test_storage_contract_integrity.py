"""Fast storage sentinels for lexical persistence and index consistency."""

from __future__ import annotations

import pytest

from rag_project.storage.vector_store import VectorStore


pytestmark = [pytest.mark.fast, pytest.mark.contract, pytest.mark.diagnostic, pytest.mark.storage]


def test_lexical_index_round_trip_is_queryable(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    store.add_documents(
        ["treatment guidance for anemia", "unrelated cardiac anatomy"],
        [
            {"document_id": "doc-a", "chunk_id": "chunk-a", "page_numbers": [1], "version_id": "v1"},
            {"document_id": "doc-b", "chunk_id": "chunk-b", "page_numbers": [2], "version_id": "v1"},
        ],
        [[0.1, 0.2], [0.2, 0.3]],
        ["vec-a", "vec-b"],
    )
    reopened = VectorStore(tmp_path / "vectors")
    result = reopened.search_lexical("anemia treatment", n_results=5)
    assert result["ids"] and result["ids"][0] == ["vec-a"], (
        "lexical index round-trip failed: expected vec-a; "
        f"received ids={result.get('ids')!r}, documents={result.get('documents')!r}"
    )
