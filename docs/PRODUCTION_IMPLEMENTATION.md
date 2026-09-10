# Production RAG implementation contract

## Scope

This release applies the complete remediation plan to the canonical medical RAG path. The objective is not to lower safety thresholds; it is to make routing, retrieval, evidence verification, abstention, and confidence semantics correct and deterministic.

## Canonical runtime

The composition root is `rag_project.application.create_rag_system`.

The authoritative answer pipeline is:

`ProductionRAGSystem.answer -> certified god-mode path -> complete_phases`

The authoritative ingestion implementation is:

`rag_project.ingestion.robust_ingestor.robust_ingest_file`

The application installs both the query/answer integrity policy and the structured production contract before constructing the service.

## P0 correctness controls

### Query isolation

Standalone questions are preserved as standalone questions. Internal diagnostics such as `Follow-up:` and `Relevant entities:` are never appended to retrieval queries.

Conversation context is used only when a genuine referential follow-up is detected. The canonicalization routine is idempotent so repeated pipeline passes cannot duplicate context.

### Entity integrity

Entity extraction is clinical/semantic rather than generic phrase extraction. Protocol labels, diagnostics, and ordinary noun phrases cannot become fake medical entities.

Entity-free summary questions therefore have zero entities and no missing-entity penalty.

### Deterministic simple routing

Simple factual/definition questions are evaluated from the canonical user query before downstream diagnostics. They are eligible for direct extractive answering and do not become multi-hop merely because an internal planner field exists.

### Abstention semantics

A controlled abstention is a structured control state. It is never sent through claim verification as though it were a medical assertion.

A blocked final gate does not manufacture a contradiction score. Contradiction is only reported when actual contradictory evidence is detected.

## P1 quality controls

### Request contract

Every answer request receives a request ID and a structured `RequestContext` containing the original query, canonical query, intent, entities, complexity, query variants, follow-up state, and retrieval constraints.

### Evidence contract

Retrieved evidence is represented as an `EvidenceBundle` with source IDs, document IDs, filenames, pages, retrieval scores, entity coverage, missing entities, and contradiction count.

### Claim-level verification

The existing claim/evidence matrix remains the hard gate. The new contract surfaces the matrix summary without weakening the existing support threshold.

### Confidence decomposition

Confidence is computed from separate retrieval, evidence-quality, entailment, entity-coverage, verification, and contradiction signals. A failed gate caps confidence but does not turn the failure into a fake contradiction.

### Multilingual behavior

Medical entity resolution continues to use the repository's semantic alias and open-set term layers. The production contract preserves those normalized entities for English/French/Arabic queries and validates them against evidence independently of generic phrase matching.

### Source provenance

Evidence source metadata remains attached to each result. The UI exposes file/page and claim-support context without placing operational metadata into the user query.

## P2 architecture controls

### Runtime ownership

The application composition root installs the integrity and production contracts. The runtime contract declares the single authoritative answer-pipeline target and canonical robust ingestion target.

### Ingestion integrity

The existing robust ingestion implementation remains responsible for:

1. file validation and hashing;
2. document classification;
3. page-level extraction;
4. semantic chunking;
5. batched embeddings;
6. incremental vector writes;
7. post-write index validation;
8. lease/heartbeat fencing;
9. atomic READY publication;
10. retirement of obsolete versions.

A new ingestion contract adds a request ID and trace envelope without changing these storage semantics.

### UI/runtime cache invalidation

The Streamlit cache version changes whenever the production contract changes, preventing an already-running process from silently retaining an older RAG system.

### Observability

The UI exposes:

- request ID and contract version;
- canonical query and follow-up state;
- complexity and intent;
- evidence hit counts and scores;
- entity coverage and missing entities;
- confidence breakdown;
- claim matrix summary;
- final verification state;
- controlled abstention reason.

## Acceptance tests

The test suite covers the exact previously observed failure and the surrounding contracts:

- standalone question remains uncontaminated;
- short standalone question does not inherit conversation history;
- genuine follow-up uses context without protocol text;
- arbitrary phrases are not medical entities;
- real medical entities survive normalization;
- French evidence can cover normalized English entities;
- simple questions do not become multi-hop;
- abstentions are not verified as medical claims;
- evidence provenance is preserved;
- failed gates do not create synthetic contradiction;
- ingestion results receive trace IDs without payload corruption;
- canonical enhancement installation is idempotent;
- all advertised production feature contracts resolve;
- the application declares one canonical answer authority.

## Local verification procedure

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall -q rag_project app.py run_dev.py
python -m pytest -q
```

For the Streamlit application, stop the previous process and start a fresh process after pulling the latest `master`. The application also changes its runtime cache version so stale cached systems are discarded automatically.

## Release rule

A release is considered implementation-complete only when the feature contract, tests, index readiness, privacy controls, and medical safety gates are all green. No safety threshold is relaxed to make the test pass.

Clinical validation and regulatory approval remain separate activities outside the software implementation contract.
