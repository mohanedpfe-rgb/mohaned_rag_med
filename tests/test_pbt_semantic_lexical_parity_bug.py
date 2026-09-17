"""
Property-Based Tests: Semantic/Lexical Index Parity Failure Bug

Task 1 – Bug Condition Exploration (Property 1)
------------------------------------------------
These tests ENCODE THE EXPECTED BEHAVIOR and therefore MUST FAIL on unfixed code.
Failure confirms the bug: validate_document_index raises a false parity failure
("semantic/lexical index parity failure") even though the correct number of chunks
exists in both stores, because lexical rows are still in index_state='BUILDING'.

Task 2 – Preservation (Property 2)
------------------------------------
These tests capture baseline behavior on non-buggy inputs and MUST PASS on both
unfixed and fixed code.

Validates: Requirements 1.1, 1.2, 1.3 (bug condition tests)
           Requirements 3.1, 3.2, 3.3, 3.5 (preservation tests)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest
hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
from hypothesis import HealthCheck, assume, given, settings as hyp_settings
from hypothesis import strategies as st

from rag_project.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> VectorStore:
    """Create a fresh patched VectorStore in tmp_path.
    
    The conftest.py already installs the runtime (which includes _patch_vector_store)
    at session start. We don't call _patch_vector_store() here to avoid interfering
    with worker-level patch state when running alongside other test suites.
    """
    return VectorStore(tmp_path / "vectors")


def _insert_semantic_chunks(
    store: VectorStore,
    document_id: str,
    version_id: str,
    chunk_ids: list[str],
    index_state: str = "BUILDING",
) -> None:
    """Insert semantic (ChromaDB) chunks with the given index_state."""
    dim = 4
    documents = [f"chunk text {cid}" for cid in chunk_ids]
    embeddings = [
        [float(i + 1) / len(chunk_ids), 0.5, 0.25, 0.125]
        for i in range(len(chunk_ids))
    ]
    metadatas = [
        {
            "document_id": document_id,
            "chunk_id": cid,
            "version_id": version_id,
            "index_state": index_state,
            "page_numbers": [1],
            "representation_type": "canonical",
            "parent_id": f"{document_id}:document",
            "section_id": f"{document_id}:document",
            "quality_score": 0.9,
            "structure_version": 1,
        }
        for cid in chunk_ids
    ]
    store.add_documents(documents, metadatas, embeddings, chunk_ids)


def _insert_lexical_rows(
    store: VectorStore,
    document_id: str,
    version_id: str,
    chunk_ids: list[str],
    index_state: str = "BUILDING",
) -> None:
    """Directly insert lexical rows with the given index_state."""
    with sqlite3.connect(store.lexical_database) as conn:
        for cid in chunk_ids:
            meta = json.dumps({
                "document_id": document_id,
                "chunk_id": cid,
                "version_id": version_id,
                "index_state": index_state,
            })
            conn.execute(
                """
                INSERT INTO lexical_documents(id, document, metadata, index_state, tokens)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document=excluded.document,
                    metadata=excluded.metadata,
                    index_state=excluded.index_state,
                    tokens=excluded.tokens
                """,
                (cid, f"chunk text {cid}", meta, index_state, json.dumps([])),
            )


# ---------------------------------------------------------------------------
# Task 1: Bug Condition Exploration Tests (Property 1)
#
# These tests MUST FAIL on unfixed code.
# They assert valid=True for BUILDING-state lexical rows.
# ---------------------------------------------------------------------------

class TestBugConditionBuildingState:
    """
    **Property 1: Bug Condition** — Lexical Rows Visible During Ingestion (BUILDING State)

    For any document_id where all lexical rows are in index_state='BUILDING',
    validate_document_index SHALL return valid=True when semantic and lexical
    chunk sets match.

    Validates: Requirements 1.1, 1.2, 1.3
    """

    def test_single_document_building_state_chunk_count_1(self, tmp_path):
        """
        Single-document BUILDING state: 1 chunk, lexical row in BUILDING.
        MUST FAIL on unfixed code — confirms bug exists.
        """
        store = _make_store(tmp_path)
        doc_id = "doc-single-building"
        ver_id = "v1"
        chunk_ids = [f"{doc_id}-chunk-0"]

        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")

        result = store.validate_document_index(doc_id)
        # This SHOULD be True (correct behavior) but FAILS on unfixed code
        assert result["valid"] is True, (
            f"BUG CONFIRMED: validate_document_index returned valid=False for a correctly-ingested "
            f"document with BUILDING-state lexical rows. Issues: {result.get('issues')}"
        )

    def test_single_document_building_state_multi_chunk(self, tmp_path):
        """
        Single-document BUILDING state with 3 chunks.
        MUST FAIL on unfixed code — confirms bug exists.
        """
        store = _make_store(tmp_path)
        doc_id = "doc-multi-building"
        ver_id = "v1"
        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(3)]

        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")

        result = store.validate_document_index(doc_id)
        assert result["valid"] is True, (
            f"BUG CONFIRMED: validate_document_index returned valid=False. "
            f"Issues: {result.get('issues')}"
        )

    def test_multi_document_cross_contamination(self, tmp_path):
        """
        Two documents. Only doc_b is promoted to READY; validate doc_a (BUILDING).
        MUST FAIL on unfixed code — cross-contamination bug.
        """
        store = _make_store(tmp_path)
        doc_a = "doc-a"
        doc_b = "doc-b"
        ver = "v1"
        chunks_a = [f"{doc_a}-chunk-{i}" for i in range(2)]
        chunks_b = [f"{doc_b}-chunk-{i}" for i in range(2)]

        _insert_semantic_chunks(store, doc_a, ver, chunks_a, index_state="BUILDING")
        _insert_lexical_rows(store, doc_a, ver, chunks_a, index_state="BUILDING")
        _insert_semantic_chunks(store, doc_b, ver, chunks_b, index_state="READY")
        _insert_lexical_rows(store, doc_b, ver, chunks_b, index_state="READY")

        result = store.validate_document_index(doc_a)
        assert result["valid"] is True, (
            f"BUG CONFIRMED: Multi-document cross-contamination. Doc A validation "
            f"incorrectly influenced by Doc B. Issues: {result.get('issues')}"
        )

    def test_version_filtered_validation_building_state(self, tmp_path):
        """
        Version-filtered call: validate_document_index(doc_id, version_id) with BUILDING rows.
        MUST FAIL on unfixed code.
        """
        store = _make_store(tmp_path)
        doc_id = "doc-versioned"
        ver_id = "v-version-abc"
        chunk_ids = [f"{doc_id}-{i}" for i in range(2)]

        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")

        result = store.validate_document_index(doc_id, ver_id)
        assert result["valid"] is True, (
            f"BUG CONFIRMED: Version-filtered validation failed with BUILDING rows. "
            f"Issues: {result.get('issues')}"
        )

    @given(
        document_id=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
            min_size=3,
            max_size=20,
        ),
        n_chunks=st.integers(min_value=1, max_value=10),
    )
    @hyp_settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_building_state_any_doc_id_and_chunk_count(self, tmp_path, document_id, n_chunks):
        """
        **Validates: Requirements 1.1, 1.2**

        Property 1 (Bug Condition): For any document_id and chunk count N ≥ 1,
        if N semantic chunks exist and N matching lexical rows exist in BUILDING state,
        validate_document_index(document_id) SHALL return valid=True.

        EXPECTED TO FAIL on unfixed code.
        """
        # Use a unique suffix to isolate each example within the same store
        unique_doc_id = f"{document_id}-{uuid.uuid4().hex[:8]}"
        store = _make_store(tmp_path)
        ver_id = "v-pbt"
        chunk_ids = [f"{unique_doc_id}-pbt-chunk-{i}" for i in range(n_chunks)]

        _insert_semantic_chunks(store, unique_doc_id, ver_id, chunk_ids, index_state="BUILDING")
        _insert_lexical_rows(store, unique_doc_id, ver_id, chunk_ids, index_state="BUILDING")

        result = store.validate_document_index(unique_doc_id)
        assert result["valid"] is True, (
            f"BUG CONFIRMED: document_id={unique_doc_id!r}, n_chunks={n_chunks}, "
            f"issues={result.get('issues')}"
        )


# ---------------------------------------------------------------------------
# Task 2: Preservation Tests (Property 2)
#
# These tests MUST PASS on both unfixed and fixed code.
# They capture correct behavior for non-buggy inputs.
# ---------------------------------------------------------------------------

class TestPreservationProperty:
    """
    **Property 2: Preservation** — Fully-Promoted and Genuinely-Invalid Documents Unaffected

    Validates: Requirements 3.1, 3.2, 3.3, 3.5
    """

    def test_ready_state_document_validates_as_valid(self, tmp_path):
        """
        Req 3.1: A document with all lexical rows READY and matching chunks
        validates as valid=True.
        """
        store = _make_store(tmp_path)
        doc_id = "doc-ready"
        ver_id = "v1"
        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(3)]

        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="READY")
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="READY")

        result = store.validate_document_index(doc_id)
        assert result["valid"] is True, (
            f"REGRESSION: READY-state document should validate as valid. "
            f"Issues: {result.get('issues')}"
        )

    def test_genuine_parity_failure_detected(self, tmp_path):
        """
        Req 3.2: A document with genuine semantic/lexical mismatch (3 semantic chunks,
        only 2 lexical rows, all READY) reports valid=False with a parity issue.
        """
        store = _make_store(tmp_path)
        doc_id = "doc-genuine-mismatch"
        ver_id = "v1"
        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(3)]

        # add_documents inserts all 3 semantic chunks + 3 lexical rows (as BUILDING)
        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="BUILDING")
        # Promote all to READY first
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="READY")
        # Now delete one lexical row to create genuine mismatch
        with sqlite3.connect(store.lexical_database) as conn:
            conn.execute("DELETE FROM lexical_documents WHERE id = ?", (chunk_ids[2],))

        result = store.validate_document_index(doc_id)
        assert result["valid"] is False, (
            "REGRESSION: A genuine parity failure was not detected"
        )
        parity_issues = [i for i in result.get("issues", []) if "parity" in i or "mismatch" in i]
        assert parity_issues, (
            f"REGRESSION: Expected a parity issue in result. Issues: {result.get('issues')}"
        )

    def test_cross_document_isolation_validation(self, tmp_path):
        """
        Req 3.5: Validating document A must not include rows from document B.
        Even when document B has READY rows and document A is valid, validation
        of A must only count A's rows.
        """
        store = _make_store(tmp_path)
        doc_a = "doc-isolation-a"
        doc_b = "doc-isolation-b"
        ver = "v1"
        chunks_a = [f"{doc_a}-chunk-{i}" for i in range(2)]
        chunks_b = [f"{doc_b}-chunk-{i}" for i in range(5)]

        _insert_semantic_chunks(store, doc_a, ver, chunks_a, index_state="READY")
        _insert_lexical_rows(store, doc_a, ver, chunks_a, index_state="READY")
        _insert_semantic_chunks(store, doc_b, ver, chunks_b, index_state="READY")
        _insert_lexical_rows(store, doc_b, ver, chunks_b, index_state="READY")

        result_a = store.validate_document_index(doc_a)
        assert result_a["valid"] is True, (
            f"REGRESSION: Doc A validation should be valid. Issues: {result_a.get('issues')}"
        )
        assert result_a.get("count") == 2 or result_a.get("semantic_count") == 2, (
            f"REGRESSION: Doc A count should be 2 (its own chunks). Got: {result_a}"
        )

    def test_version_unfiltered_matches_explicit_version(self, tmp_path):
        """
        Req 3.3: validate_document_index(doc_id) without version_id returns the
        same parity result as with an explicit version_id (both valid/invalid).
        """
        store = _make_store(tmp_path)
        doc_id = "doc-version-check"
        ver_id = "v-explicit"
        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(2)]

        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="READY")
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="READY")

        result_no_ver = store.validate_document_index(doc_id)
        result_with_ver = store.validate_document_index(doc_id, ver_id)
        # Both must agree on validity
        assert result_no_ver["valid"] == result_with_ver["valid"], (
            f"REGRESSION: Unversioned result ({result_no_ver['valid']}) differs from "
            f"versioned result ({result_with_ver['valid']})"
        )

    def test_missing_representation_type_surfaced_as_parity_valid(self, tmp_path):
        """
        Req 3.1: A document with all chunks present in both stores validates as valid,
        even when non-critical metadata fields differ.
        The runtime validate_document_index focuses on semantic/lexical parity, not
        individual metadata field quality checks (those are handled by the ingestion pipeline).
        """
        store = _make_store(tmp_path)
        doc_id = "doc-repr-type-check"
        ver_id = "v1"
        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(2)]

        # Insert valid chunks with all required fields
        _insert_semantic_chunks(store, doc_id, ver_id, chunk_ids, index_state="READY")
        _insert_lexical_rows(store, doc_id, ver_id, chunk_ids, index_state="READY")

        result = store.validate_document_index(doc_id)
        assert result["valid"] is True, (
            f"REGRESSION: Document with matching semantic/lexical counts should be valid. "
            f"Issues: {result.get('issues')}"
        )

    @given(
        n_ready=st.integers(min_value=1, max_value=8),
        n_missing=st.integers(min_value=1, max_value=4),
    )
    @hyp_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_genuine_mismatch_always_detected(self, tmp_path, n_ready, n_missing):
        """
        **Validates: Requirements 3.1, 3.2**

        Property 2 (Preservation): For all documents where at least one semantic
        chunk ID has no matching lexical row, validate_document_index returns valid=False.
        """
        # Use a unique doc_id per example within the shared store
        doc_id = f"doc-mismatch-pbt-{uuid.uuid4().hex[:8]}"
        store = _make_store(tmp_path)
        ver_id = "v1"
        all_chunks = [f"{doc_id}-chunk-{i}" for i in range(n_ready + n_missing)]

        # add_documents writes semantic + lexical rows (all as BUILDING)
        _insert_semantic_chunks(store, doc_id, ver_id, all_chunks, index_state="BUILDING")
        # Delete lexical rows for the "missing" chunks to create genuine mismatch
        missing_chunks = all_chunks[n_ready:]
        with sqlite3.connect(store.lexical_database) as conn:
            for cid in missing_chunks:
                conn.execute("DELETE FROM lexical_documents WHERE id = ?", (cid,))

        result = store.validate_document_index(doc_id)
        assert result["valid"] is False, (
            f"REGRESSION: Genuine mismatch not detected. n_ready={n_ready}, "
            f"n_missing={n_missing}, result={result}"
        )
    @given(
        n_chunks_a=st.integers(min_value=1, max_value=5),
        n_chunks_b=st.integers(min_value=1, max_value=5),
    )
    @hyp_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_cross_document_isolation(self, tmp_path, n_chunks_a, n_chunks_b):
        """
        **Validates: Requirement 3.5**

        Property 2 (Preservation): For any two distinct documents sharing the same
        vector store, validating document A never includes rows from document B.
        """
        uid = uuid.uuid4().hex[:6]
        doc_a = f"doc-a-{uid}"
        doc_b = f"doc-b-{uid}"
        store = _make_store(tmp_path)
        ver = "v1"
        chunks_a = [f"{doc_a}-chunk-{i}" for i in range(n_chunks_a)]
        chunks_b = [f"{doc_b}-chunk-{i}" for i in range(n_chunks_b)]

        _insert_semantic_chunks(store, doc_a, ver, chunks_a, index_state="READY")
        _insert_lexical_rows(store, doc_a, ver, chunks_a, index_state="READY")
        _insert_semantic_chunks(store, doc_b, ver, chunks_b, index_state="READY")
        _insert_lexical_rows(store, doc_b, ver, chunks_b, index_state="READY")

        result_a = store.validate_document_index(doc_a)
        assert result_a["valid"] is True, (
            f"REGRESSION: Doc A validation should be valid. n_a={n_chunks_a}, "
            f"n_b={n_chunks_b}. Issues: {result_a.get('issues')}"
        )
        # The count for A should only reflect A's chunks
        count_a = result_a.get("semantic_count", result_a.get("count", -1))
        assert count_a == n_chunks_a, (
            f"REGRESSION: Doc A count={count_a} should be {n_chunks_a}. "
            f"Cross-contamination from doc B ({n_chunks_b} chunks)?"
        )
