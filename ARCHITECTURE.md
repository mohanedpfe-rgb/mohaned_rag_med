# BookRAG architecture contract

This project is intentionally designed as a modular, single-node RAG service that can grow without forcing a rewrite of the ingestion, retrieval, or generation stack.

## Layering

```text
UI / scripts
    ↓
Application composition + runtime policy
    ↓
RAG application service
    ↓
Ingestion | Retrieval | Generation | Citations
    ↓
Storage / external adapters (SQLite, Chroma, Ollama)
```

The UI must not own persistence rules, ingestion state transitions, embedding lifecycle, or retrieval algorithms. Those concerns belong below the application boundary.

## Long-lived invariants

1. A document version is immutable once published. Builds are isolated and become queryable only after validation.
2. SQLite is the durable source for ingestion state and full page text; Chroma and lexical data are published with explicit readiness state and recoverable cleanup.
3. Physical PDF page numbers remain separate from printed page labels so citations and checkpoints cannot be corrupted by document pagination.
4. Embedding identity includes model identity, endpoint/profile, and vector dimension. A changed embedding contract must never silently reuse an old index.
5. External model calls are bounded. A timeout must bound the operation itself, not merely the retry delay around it.
6. Cancellation, leases, and final writes are ownership-aware. A worker that loses ownership must not publish terminal state.
7. Ingestion is bounded-memory: pages/chunks are processed in batches instead of retaining the whole document and all vectors in RAM.
8. OCR is enabled by default on the i5/16GB profile and is automatically used for scanned/image-heavy pages when the page is judged to need it; OCR failures remain visible in page state.
9. Tables and figure captions are represented as first-class searchable evidence units. Scanned tables receive a bounded OCR/layout recovery attempt instead of being silently flattened into ordinary prose.
10. Every child chunk retains document hierarchy context (chapter, section, parent, page, quality, OCR state, table/figure identity) in vector metadata.
11. A separate hierarchy index stores book, chapter, and section aggregate vectors so document-level retrieval is not forced to operate only on child chunks.
12. Retrieval is multilingual-aware for English, French, and Arabic, and generation is grounded in retrieved evidence with refusal on insufficient alignment.
13. Every production entry point must pass through the composition/runtime boundary so safety and recovery policies are installed before services are constructed.

## Dependency direction

Feature modules may depend on lower-level utilities/adapters, but storage adapters must not import UI modules, and the UI must not implement storage semantics. Cross-cutting runtime policies belong in the runtime boundary and should be migrated into the owning service when the next structural refactor touches that service.

## Scalability target

The reference deployment is a single machine with roughly 16GB RAM and a CPU-first workload. Concurrency is deliberately bounded around expensive model operations. Horizontal scaling should be introduced only after storage ownership is separated behind explicit interfaces; otherwise multiple writers can turn operational complexity into correctness bugs.

## Maintainability target

Changes should be localized by subsystem: parsing, OCR, chunking, embedding, indexing, retrieval, reranking, generation, or UI. New integrations should be adapters rather than conditionals spread through the RAG orchestration layer. Tests should exercise contracts and failure modes, not only happy-path answers.

## Architectural debt register

The current codebase contains a legacy `RAGSystem` implementation that is still large. Runtime hardening currently surrounds parts of that legacy surface. This is intentional transitional architecture: correctness and recovery were stabilized first. The next refactor should move the highest-value runtime wrappers into explicit service methods, then delete the corresponding compatibility patches. Until then, `rag_project/runtime.py` is the single composition point and should not be bypassed by production entry points.
