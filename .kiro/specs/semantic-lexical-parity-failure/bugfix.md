# Bugfix Requirements Document

## Introduction

Every document ingestion fails with a `RuntimeError` citing "semantic/lexical index parity failure"
even when the numerical counts match (e.g., "expected 1, found 1"). This blocks ~80 % of all test
cases because any test that ingests a PDF and asserts `status == "success"` / `status == "READY"`
is guaranteed to fail.

The failure originates in the monkey-patched `validate_document_index` installed by
`rag_project/intelligence/deep_pdf_contract.py`. That override queries the lexical store with
`WHERE index_state = 'READY'` but **without a `document_id` filter**. During the ingestion
pipeline, `add_documents` writes lexical records with `index_state = 'BUILDING'`; the state is
only promoted to `'READY'` *after* the validation gate passes. Consequently the parity check
always finds zero READY lexical rows for the document under validation, falsely concludes the
semantic and lexical indexes are mismatched, and raises the error — regardless of how many chunks
were actually written.

---

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a document is ingested and `validate_document_index` is called before
`set_version_index_state("READY")` completes THEN the system raises
`RuntimeError: FAILED_INDEXING: committed index validation failed … semantic/lexical index parity failure`
even though the correct number of chunks exists in both stores.

1.2 WHEN the patched `validate_document_index` queries the lexical store for parity THEN the
system uses the filter `WHERE index_state = 'READY'` with no `document_id` constraint, causing
it to miss all just-written lexical rows whose `index_state` is still `'BUILDING'`.

1.3 WHEN `embedding_test_mode=True` is active during testing THEN the system still reaches the
deep-contract parity check and raises the same failure, blocking every test that expects
`status == "success"` or `status == "READY"`.

### Expected Behavior (Correct)

2.1 WHEN `validate_document_index` is called for a given `document_id` and `version_id` during
the ingestion pipeline THEN the system SHALL match lexical rows by `document_id` (and optionally
by `chunk_id` join key) rather than by `index_state`, so that rows in `'BUILDING'` state are
included in the parity comparison.

2.2 WHEN the lexical parity check executes THEN the system SHALL scope its lexical query to the
specific `document_id` being validated, returning counts and chunk-id sets that reflect all chunks
written for that document regardless of their current `index_state`.

2.3 WHEN all semantic chunk IDs for the document have a matching lexical record (in any
`index_state`) THEN the system SHALL report `valid = True` and allow ingestion to proceed to the
`READY` state transition.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a document that was previously ingested successfully is re-validated after
`set_version_index_state("READY")` THEN the system SHALL CONTINUE TO report `valid = True` and
return the correct chunk count.

3.2 WHEN a document has a genuine semantic/lexical mismatch (chunks present in the semantic
vector store but absent from the lexical SQLite store) THEN the system SHALL CONTINUE TO report
`valid = False` with an appropriate issue message.

3.3 WHEN `validate_document_index` is called without a `version_id` THEN the system SHALL
CONTINUE TO validate all chunks for the document and report the same parity result as before the
fix.

3.4 WHEN an ingestion is invoked with a real embedding model (not test mode) THEN the system SHALL
CONTINUE TO complete the full ingestion pipeline and reach `status == "READY"` without regression.

3.5 WHEN two different documents are indexed in the same vector store THEN the system SHALL
CONTINUE TO validate each document independently, and a parity failure for one document SHALL NOT
affect the validation result of the other.

---

## Bug Condition (Pseudocode)

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type IngestionContext
  OUTPUT: boolean

  // Bug fires when the parity check runs against a document whose lexical
  // rows have not yet been promoted to 'READY', AND the override query
  // omits the document_id filter.
  RETURN X.lexical_index_state = 'BUILDING'
     AND deep_contract_patch_active(X.vector_store)
     AND lexical_query_lacks_document_id_filter(X.vector_store.validate_document_index)
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Fix Checking — parity validation succeeds during in-progress ingestion
FOR ALL X WHERE isBugCondition(X) DO
  result ← validate_document_index'(X.document_id, X.version_id)
  ASSERT result["valid"] = True
  ASSERT result["count"] = X.expected_chunk_count
  ASSERT "semantic/lexical index parity failure" NOT IN result["issues"]
END FOR
```

### Preservation Property

```pascal
// Property: Preservation Checking — non-buggy ingestions are unaffected
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT validate_document_index(X) = validate_document_index'(X)
END FOR
```
