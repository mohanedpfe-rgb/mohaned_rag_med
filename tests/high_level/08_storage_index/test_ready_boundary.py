from __future__ import annotations

import pytest

from rag_project.retrieval.ready_only_retriever import ReadyOnlyRetriever


@pytest.mark.high_level
def test_storage__caller_cannot_override_ready_only_visibility_boundary():
    ready_where = ReadyOnlyRetriever._ready_where({"index_state": "BUILDING", "document_id": "doc-1"})
    assert ready_where.get("$and")
    clauses = ready_where["$and"]
    assert {"index_state": "READY"} in clauses
    assert {"document_id": "doc-1"} in clauses
    assert {"index_state": "BUILDING"} not in clauses


@pytest.mark.high_level
def test_storage__empty_filter_still_injects_ready_state_constraint():
    assert ReadyOnlyRetriever._ready_where(None) == {"index_state": "READY"}
    assert ReadyOnlyRetriever._ready_where({}) == {"index_state": "READY"}
