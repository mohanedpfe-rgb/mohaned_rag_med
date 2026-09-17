# Semantic/Lexical Index Parity Failure Bugfix Design

## Overview

During document ingestion, `VectorStore.validate_document_index` always reports a
"semantic/lexical index parity failure" even for correctly-ingested documents.

The root cause is a monkey-patch installed by
`rag_project/intelligence/deep_pdf_contract.py` inside `_patch_vector_store()`.
The patched `validate_document_index` queries the lexical index with two defects:

1. **No `document_id` filter** — the query reads lexical rows across *all* documents,
   not just the one being validated.
2. **Premature `index_state='READY'` filter** — lexical rows are written as
   `'BUILDING'` during ingestion and promoted to `'READY'` only *after* validation
   passes, creating a circular dependency: validation requires `'READY'` rows, but
   rows are only promoted once validation passes.

The base implementation in `rag_project/storage/vector_store.py` is correct — it
scopes rows by `json_extract(metadata, '$.document_id') = ?` with no state filter.
The fix (Option A) corrects the patched query in-place while preserving the richer
metadata field checks the patch adds (representation_type, parent_id, quality_score,
table_id, figure_id, etc.) that the base implementation does not perform.

---

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — when `validate_document_index`
  is called during or immediately after ingestion, the lexical query returns zero rows,
  causing a false parity failure.
- **Property (P)**: The desired behavior — `validate_document_index` returns `valid=True`
  for a correctly-ingested document whose lexical rows exist in any `index_state`.
- **Preservation**: All behavior of `validate_document_index` for documents that are
  genuinely invalid (real parity failures, missing fields, bad embeddings) must remain
  unchanged; valid documents in a fully-promoted (`READY`) state must still pass.
- **`validate_document_index`**: The method on `VectorStore`
  (in `rag_project/intelligence/deep_pdf_contract.py` when patched) that checks
  semantic and lexical index consistency for a given `document_id`.
- **`index_state`**: A metadata field on every lexical row indicating ingestion lifecycle:
  `'BUILDING'` (written during ingestion), `'READY'` (promoted after validation),
  `'ERROR'` (failed).
- **monkey-patch**: The runtime replacement of `VectorStore.validate_document_index`
  performed by `_patch_vector_store()` in `deep_pdf_contract.py` at import time.

---

## Bug Details

### Bug Condition

The bug manifests when `validate_document_index` is called for a document whose lexical
rows are in `index_state='BUILDING'` (i.e., during or right after ingestion). The patched
query ignores both the target `document_id` and lexical rows that are still `'BUILDING'`,
so it always finds zero matching rows for the document being validated and incorrectly
reports a parity failure.

**Formal Specification:**

```
FUNCTION isBugCondition(document_id, version_id)
  INPUT:  document_id  — the document being validated
          version_id   — optional version / content-hash
  OUTPUT: boolean

  lexical_rows_for_doc := SELECT * FROM lexical_documents
                          WHERE json_extract(metadata, '$.document_id') = document_id

  RETURN  lexical_rows_for_doc IS NOT EMPTY
          AND ALL rows IN lexical_rows_for_doc HAVE index_state = 'BUILDING'
          -- i.e., the document has been ingested but not yet promoted
END FUNCTION
```

### Examples

- **Normal ingestion** — a 50-chunk PDF is ingested; all lexical rows land as `'BUILDING'`.
  Validation is called. The patched query returns 0 rows (wrong), the base would return 50.
  Result: `"semantic/lexical index parity failure"` — **false positive bug**.

- **Multi-document store** — two documents exist; both have some `'READY'` rows. Validation
  is called for document A. The patched query returns rows from both A *and* B with
  `index_state='READY'`, inflating the lexical set — **wrong scope bug**.

- **Post-promotion (READY)** — after a successful promotion cycle, rows are `'READY'`.
  Validation passes. This is the only scenario the patched code handles correctly, which
  is why the bug is not caught in all environments.

- **Edge case — empty document** — zero semantic chunks. The patched code accidentally
  passes (`semantic_keys` is empty, `issubset` of anything is `True`) but for the wrong
  reason; fixed code must handle this correctly too.

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**

- Documents with a genuine semantic/lexical mismatch (different chunk counts, missing
  chunk IDs) must still be reported as invalid.
- The extra metadata field checks added by the patch (representation_type, parent_id,
  section_id, quality_score, table_id, figure_id, structure_version) must continue to
  run and report issues.
- `valid=True` must still require `selected_ids` to be non-empty (no empty-document
  false positives).
- All callers that depend on the returned dict keys (`document_id`, `count`, `valid`,
  `issues`) must continue to receive the same structure.

**Scope:**

All inputs that do NOT match the bug condition (lexical rows already promoted to `'READY'`,
genuinely invalid documents, documents with no lexical rows at all) should be completely
unaffected by this fix. This includes:

- Documents that have already completed the full ingestion-and-promotion lifecycle.
- Documents with real structural issues (bad embeddings, missing fields).
- Any call path that does not go through the patched `validate_document_index`.

---

## Hypothesized Root Cause

Based on the confirmed bug description, the causes are:

1. **Missing `document_id` scope** — the query
   `SELECT id, metadata FROM lexical_documents WHERE index_state='READY'`
   reads every READY row in the entire database. For a multi-document store this returns
   rows belonging to other documents, and for a freshly-ingested document it returns
   nothing from that document.

2. **Incorrect lifecycle assumption** — the patch assumes lexical rows are already
   `'READY'` at the time `validate_document_index` is called. The actual ingestion flow
   calls validation *before* calling `set_version_index_state(..., 'READY')`, so rows
   are still `'BUILDING'` at that point.

3. **Circular dependency** — the `index_state='READY'` guard creates a deadlock:
   validation must pass before rows are promoted, but rows must be `'READY'` before
   validation can pass.

4. **Divergence from the base implementation** — the base `VectorStore.validate_document_index`
   uses `json_extract(metadata, '$.document_id') = ?` with no state filter; the patch
   introduced both defects when it was written, without mirroring the correct query from
   the base class.

---

## Correctness Properties

Property 1: Bug Condition — Lexical Rows Visible During Ingestion

_For any_ `document_id` where the bug condition holds (lexical rows exist in
`index_state='BUILDING'` for that document), the fixed `validate_document_index` SHALL
return `valid=True` when the semantic and lexical chunk sets match, regardless of the
`index_state` of the lexical rows.

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation — Genuine Parity Failures Still Detected

_For any_ `document_id` where the bug condition does NOT hold (lexical rows are already
`'READY'`, or there is a real mismatch), the fixed `validate_document_index` SHALL
produce the same `valid` result and issue list as the original patched function would
produce on a fully-promoted document, preserving all metadata field validation and
parity-failure detection.

**Validates: Requirements 3.1, 3.2, 3.3**

---

## Fix Implementation

### Changes Required

**File:** `rag_project/intelligence/deep_pdf_contract.py`

**Function:** `validate_document_index` (defined inside `_patch_vector_store()`)

**Specific Changes:**

1. **Replace the lexical query** — change:
   ```python
   rows = connection.execute(
       "SELECT id, metadata FROM lexical_documents WHERE index_state='READY'"
   ).fetchall()
   ```
   to:
   ```python
   rows = connection.execute(
       "SELECT id, metadata FROM lexical_documents "
       "WHERE json_extract(metadata, '$.document_id') = ?",
       (str(document_id),)
   ).fetchall()
   ```

2. **Remove the `index_state='READY'` guard from parity logic** — the `lexical_ids`
   set is currently built from rows already filtered to `'READY'`. After the query fix,
   all rows for the document are returned; no additional state filtering should be
   applied before building `lexical_ids` / the chunk-key set.

3. **Update the `semantic_keys.issubset(lexical_ids)` check** — since `lexical_ids`
   is now a chunk-key set scoped to the target document, the subset check is correct
   without further modification.

4. **Preserve all metadata field validation** — the per-chunk loop that checks
   `representation_type`, `parent_id`, `section_id`, `quality_score`, `table_id`,
   `figure_id`, `structure_version`, embedding validity, and document text must remain
   entirely unchanged.

5. **No changes to call sites** — `validate_document_index` is called by the ingestion
   pipeline; the fix is purely internal to the patched method body.

---

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that
demonstrate the bug on unfixed code, then verify the fix works correctly and preserves
existing behavior.

### Exploratory Bug Condition Checking

**Goal:** Surface counterexamples that demonstrate the bug *before* implementing the fix.
Confirm or refute the root cause analysis. If refuted, re-hypothesize.

**Test Plan:** Write tests that create a `VectorStore`, install the `deep_pdf_contract`
patch, ingest a document (leaving lexical rows as `'BUILDING'`), then call
`validate_document_index`. Assert that `valid=True` — these assertions will fail on
unfixed code, exposing the bug.

**Test Cases:**

1. **Single-document BUILDING state** — ingest one document; lexical rows remain
   `'BUILDING'`; call `validate_document_index(doc_id)` — will fail on unfixed code.
2. **Multi-document cross-contamination** — ingest two documents, promote only the
   second to `'READY'`; validate the first — will fail on unfixed code (wrong scope).
3. **Version-filtered validation** — call `validate_document_index(doc_id, version_id)`
   with rows in `'BUILDING'` — will fail on unfixed code.
4. **Edge case — out-of-range key** — ingest a document with 3 chunks; validate; key 9
   has no corresponding lexical row (may fail on unfixed code depending on state).

**Expected Counterexamples:**

- `validate_document_index` returns `valid=False` with issue
  `"semantic/lexical index parity failure"` even though the document is correctly ingested.
- Possible causes confirmed by analysis: missing `document_id` filter, premature
  `index_state='READY'` guard.

### Fix Checking

**Goal:** Verify that for all inputs where the bug condition holds, the fixed function
produces the expected behavior.

**Pseudocode:**
```
FOR ALL (document_id, version_id) WHERE isBugCondition(document_id, version_id) DO
  result := validate_document_index_fixed(document_id, version_id)
  ASSERT result["valid"] == True
  ASSERT "semantic/lexical index parity failure" NOT IN result["issues"]
END FOR
```

### Preservation Checking

**Goal:** Verify that for all inputs where the bug condition does NOT hold, the fixed
function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL (document_id, version_id) WHERE NOT isBugCondition(document_id, version_id) DO
  ASSERT validate_document_index_original(document_id, version_id)
      == validate_document_index_fixed(document_id, version_id)
END FOR
```

**Testing Approach:** Property-based testing is recommended for preservation checking
because:
- It generates many random document states and chunk configurations automatically.
- It catches edge cases (empty documents, single-chunk documents, documents with mixed
  `index_state` values) that manual unit tests might miss.
- It provides strong guarantees that fully-promoted documents are unaffected.

**Test Plan:** Observe behavior on unfixed code for fully-promoted (`'READY'`) documents
and genuinely-invalid documents, then write property-based tests capturing that behavior.

**Test Cases:**

1. **READY state preservation** — documents with all lexical rows `'READY'` must
   continue to validate as they did before the fix.
2. **Genuine parity failure preservation** — documents with mismatched semantic/lexical
   chunk sets must still report `valid=False` after the fix.
3. **Metadata field validation preservation** — documents with missing `parent_id`,
   bad `representation_type`, or invalid embeddings must still report their issues.

### Unit Tests

- Test `validate_document_index` with a single document whose lexical rows are in
  `'BUILDING'` state — expect `valid=True` after fix.
- Test with a multi-document store — only rows for the target document should be
  counted, not rows from other documents.
- Test the edge case where `document_id` has zero semantic chunks — expect `valid=False`
  (non-empty check must still hold).
- Test that a real parity failure (semantic chunk without a matching lexical row) is
  still detected correctly after the fix.

### Property-Based Tests

- Generate random sets of chunk IDs and verify that the fixed function returns
  `valid=True` whenever semantic and lexical chunk sets match, regardless of
  `index_state`.
- Generate random multi-document stores and verify that validation of document A is
  never affected by the presence of document B's rows.
- Generate random metadata configurations with intentional field omissions and verify
  that the metadata validation issues are preserved identically after the fix.

### Integration Tests

- Full ingestion of a PDF through the patched pipeline; verify that
  `validate_document_index` returns `valid=True` before `set_version_index_state`
  promotes rows to `'READY'`.
- Full ingestion followed by promotion; verify that `validate_document_index` still
  returns `valid=True` after promotion.
- Two-document ingestion; validate each document independently; verify no
  cross-contamination in issue lists.
