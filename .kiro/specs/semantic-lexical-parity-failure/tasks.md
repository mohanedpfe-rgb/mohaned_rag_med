# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Lexical Rows Visible During Ingestion (BUILDING State)
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug: `validate_document_index` raises a parity failure even though the correct number of chunks exists in both stores
  - **Scoped PBT Approach**: Scope the property to the concrete failing scenario — documents whose lexical rows are in `index_state='BUILDING'` (i.e., mid-ingestion) with any valid chunk count
  - Setup: create a VectorStore, install the `deep_pdf_contract` patch (import `_patch_vector_store` from `rag_project/intelligence/deep_pdf_contract.py`), ingest a document leaving lexical rows as `'BUILDING'`, then call `validate_document_index(document_id)`
  - Bug condition from design: `isBugCondition(X)` where `X.lexical_index_state = 'BUILDING'` AND `deep_contract_patch_active(X.vector_store)` AND `lexical_query_lacks_document_id_filter(...)`
  - Property-based: generate arbitrary `document_id` values and chunk counts (N ≥ 1); for each, insert N semantic chunks and N lexical rows with `index_state='BUILDING'`, assert `validate_document_index(document_id)["valid"] == True`
  - Concrete sub-cases to cover:
    - Single-document BUILDING state: one document, all lexical rows `'BUILDING'`, chunk count = 1
    - Multi-document cross-contamination: two documents, only second promoted to `'READY'`; validate first — should return `valid=True`, not cross-contaminate
    - Version-filtered validation: call `validate_document_index(doc_id, version_id)` with rows in `'BUILDING'` — should return `valid=True`
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (confirms bug — `valid=False` with "semantic/lexical index parity failure")
  - Document counterexamples found (e.g., `validate_document_index("doc-abc", None)` returns `{"valid": False, "issues": ["semantic/lexical index parity failure"]}` even with 5 matching chunks)
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Fully-Promoted and Genuinely-Invalid Documents Unaffected
  - **IMPORTANT**: Follow observation-first methodology
  - Observe UNFIXED code behavior for NON-buggy inputs (cases where `isBugCondition` returns False):
    - Observe: `validate_document_index(doc_id)` for a document with all lexical rows already `'READY'` returns `valid=True`
    - Observe: `validate_document_index(doc_id)` for a document with a genuine semantic/lexical mismatch (e.g., 3 semantic chunks, only 2 lexical rows, all `'READY'`) returns `valid=False` with "semantic/lexical index parity failure"
    - Observe: `validate_document_index(doc_id)` for a document with a missing `representation_type` field returns `valid=False` with a metadata-related issue message
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements:
    - Property: for all documents where all lexical rows are `'READY'` and semantic ↔ lexical chunk sets match, `validate_document_index` returns `valid=True` (Requirement 3.1)
    - Property: for all documents where at least one semantic chunk ID has no matching lexical row, `validate_document_index` returns `valid=False` (Requirement 3.2)
    - Property: for any two distinct documents sharing the same vector store, validating document A never includes rows from document B (Requirement 3.5)
    - Property: calling `validate_document_index` without a `version_id` returns the same parity result as with an explicit version (Requirement 3.3)
    - Property: all metadata field checks (representation_type, parent_id, section_id, quality_score, table_id, figure_id, structure_version) continue to surface issues when fields are missing or invalid (Requirement 3.1, 3.2)
  - Property-based testing is recommended for stronger preservation guarantees — generates many random document states and chunk configurations
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [ ] 3. Fix the lexical query scope and return dict in `validate_document_index`

  - [ ] 3.1 Replace the broken lexical query in `deep_pdf_contract.py`
    - File: `rag_project/intelligence/deep_pdf_contract.py`
    - Function: `validate_document_index` defined inside `_patch_vector_store()`
    - Replace:
      ```python
      rows = connection.execute("SELECT id, metadata FROM lexical_documents WHERE index_state='READY'").fetchall()
      ```
      with:
      ```python
      rows = connection.execute(
          "SELECT id, metadata FROM lexical_documents "
          "WHERE json_extract(metadata, '$.document_id') = ?",
          (str(document_id),)
      ).fetchall()
      ```
    - This removes the `index_state='READY'` filter (circular dependency) and adds correct `document_id` scope
    - Do NOT add any post-query `index_state` filtering when building `lexical_ids` — all rows for the document should be included regardless of state
    - Do NOT modify the per-chunk metadata validation loop (representation_type, parent_id, section_id, quality_score, table_id, figure_id, structure_version, embedding validity) — leave it entirely unchanged
    - _Bug_Condition: isBugCondition(X) where X.lexical_index_state = 'BUILDING' AND deep_contract_patch_active(X.vector_store) AND lexical_query_lacks_document_id_filter(...)_
    - _Expected_Behavior: result["valid"] = True AND result["count"] = expected_chunk_count AND "semantic/lexical index parity failure" NOT IN result["issues"]_
    - _Preservation: Documents with genuine mismatches still report valid=False; metadata field checks are untouched; multi-document stores validate each document in isolation_
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.2 Extend the return dict with `semantic_count` and `lexical_count` keys
    - In the same `validate_document_index` function, update the final `return` statement to include `semantic_count` and `lexical_count` alongside the existing keys
    - New return structure:
      ```python
      return {
          "document_id": document_id,
          "count": len(selected_ids),
          "semantic_count": len(selected_ids),
          "lexical_count": len(lexical_ids),
          "valid": valid,
          "issues": issues,
      }
      ```
    - This matches the base `VectorStore.validate_document_index` signature so callers accessing `result["semantic_count"]` no longer get a `KeyError`
    - Existing keys (`document_id`, `count`, `valid`, `issues`) are preserved unchanged — no call-site changes needed
    - _Requirements: 2.3, 3.1_

  - [ ] 3.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Lexical Rows Visible During Ingestion (BUILDING State)
    - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
    - The test from task 1 encodes the expected behavior: `validate_document_index` returns `valid=True` for documents whose lexical rows are in `'BUILDING'` state
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed — no more "semantic/lexical index parity failure" false positive)
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Fully-Promoted and Genuinely-Invalid Documents Unaffected
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in READY-state validation, genuine-failure detection, metadata field checks, and cross-document isolation)
    - Confirm all tests still pass after fix (no regressions)

- [ ] 4. Run integration test suites and confirm all pass
  - Run `pytest tests/test_post_index_publication_phases.py -x`
  - Run `pytest tests/test_hardening.py -x`
  - Run `pytest tests/test_production_transaction_smoke.py -x`
  - All three suites must pass with zero failures
  - Ensure all tests pass; ask the user if questions arise
  - _Requirements: 1.1, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5_
