# BookRAG architecture contract

This project is a modular, single-node medical RAG service optimized for CPU-first local execution around 16GB RAM. The ingestion path preserves full page text, **document hierarchy**, document-global structural ancestry, alternate table/figure representations, and build-aware publication semantics.

## Layering

```text
UI / scripts
    ↓
Application composition + runtime policy
    ↓
RAG application service
    ↓
Document representation
    ├── native PDF text
    ├── conditional OCR
    ├── heading / hierarchy inference
    ├── table representations
    ├── figure/caption representations
    └── durable page-structure sidecar
    ↓
Ingestion | Retrieval | Generation | Citations
    ↓
Storage / external adapters (SQLite, structure SQLite, Chroma, Ollama)
```

The UI must not own persistence rules, ingestion state transitions, embedding lifecycle, or retrieval algorithms.

## Long-lived invariants

1. A document version is immutable once published. Builds are isolated with build identifiers and are queryable only after structural, semantic and lexical validation.
2. SQLite is the durable source for ingestion state and full page text; the structure sidecar is the durable source for page-level representation needed to audit/rebuild hierarchy; Chroma and lexical data are published with explicit readiness state.
3. Physical PDF page numbers remain separate from printed page labels so citations and checkpoints cannot be corrupted by document pagination.
4. Embedding identity includes model identity, endpoint/profile, and vector dimension. File content hash is stored separately from the canonical ingestion `version_id`.
5. External model calls are bounded. A timeout must bound the operation itself, not merely the retry delay around it.
6. Cancellation, leases, and final writes are ownership-aware. A worker that loses ownership must not publish terminal state.
7. Ingestion is bounded-memory: pages/chunks are processed in batches instead of retaining the whole document and all vectors in RAM.
8. OCR is enabled in the i5/16GB profile and remains conditional per page. Final merged OCR text is re-analyzed for captions and table-like regions.
9. Tables and figure captions are first-class searchable evidence units. Scanned pages receive a bounded OCR/layout-recovery attempt instead of being silently treated as ordinary prose.
10. Every child representation retains document-global hierarchy context: chapter, section, parent, page, quality, OCR state, and table/figure identity.
11. Chapter/section identity is document-global and survives page boundaries. Dedicated whole-book vectors are not required for correctness; hierarchical child representations are the canonical retrieval units.
12. Retrieval is representation-aware: table/figure queries receive targeted structural boosts, and context assembly preserves evidence diversity before applying per-document limits.
13. Every production entry point must pass through the composition/runtime boundary so representation, storage, safety and recovery policies cannot be silently bypassed.

## Dependency direction

Feature modules may depend on lower-level utilities/adapters, but storage adapters must not import UI modules, and the UI must not implement storage semantics. Cross-cutting runtime policies belong in the runtime boundary and should be migrated into owning service methods when the next structural refactor touches those services.

## Scalability target

The reference deployment is a single machine with roughly 16GB RAM and a CPU-first workload. Concurrency is deliberately bounded around expensive model operations. Visual embeddings remain optional so figure/caption search does not make the default laptop profile unusable.

## Architectural debt register

The legacy `RAGSystem` surface remains transitional. Runtime installers are the composition boundary. The deep PDF contract is installed from that boundary so parsing, structure, versioning, validation and retrieval policies are active for production entry points.
