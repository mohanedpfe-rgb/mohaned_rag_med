# BookRAG architecture contract

BookRAG is a modular, single-node medical RAG service optimized for CPU-first local execution around 16GB RAM. The ingestion path preserves full page text, document hierarchy, document-global structural ancestry, alternate table/figure representations, and build-aware publication semantics.

## Runtime boundary

`app.py` is the thin presentation/orchestration entrypoint. It owns only Streamlit composition, UI finish ordering, supervisor startup, and runtime-cache invalidation.

The production composition boundary is `rag_project.composition.prepare_runtime`. It owns local environment loading, bounded runtime normalization, and authoritative installer ordering. This module intentionally has no UI dependency so the production runtime policy can be tested and reused independently of Streamlit.

The UI security capture boundary is `rag_project.app.ui_security_boundary.install`. It owns upload/path/URL validation capture and presentation-side evidence/intelligence panel wiring. Security policy itself remains in `rag_project.security`; this module only composes it around the UI surface.

The canonical application service remains `rag_project.application.MedEvidenceProductionRAGSystem`, and the canonical answer authority remains `rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer`.

## Layering

```text
UI / scripts
    ↓
Presentation/orchestration entrypoint (`app.py`)
    ↓
UI security capture (`rag_project.app.ui_security_boundary`)
    ↓
Composition root (`rag_project.composition`)
    ↓
Application service (`rag_project.application`)
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
14. The presentation entrypoint must remain thin: runtime installers, environment normalization, canonical-service selection, and security implementation must not be reimplemented in `app.py`.
15. Composition modules must not depend on presentation/UI modules, preventing a reverse dependency from infrastructure into Streamlit.
16. UI security capture is isolated from the application entrypoint and delegates policy decisions to lower-level security services.

## Dependency direction

Feature modules may depend on lower-level utilities/adapters, but storage adapters must not import UI modules, the UI must not implement storage semantics, and the composition root must not depend on presentation modules. UI security capture may depend on the UI and security adapter surfaces but must not become a storage or domain layer. Cross-cutting runtime policies belong in the composition/runtime boundary and should be migrated into owning service methods when the next structural refactor touches those services.

## Executable architecture quality gate

Architecture is enforced before the test suite by `scripts/architecture_gate.py`. It is standard-library-only and therefore safe to run before optional runtime imports. The gate checks the thin `app.py` ceiling, composition/UI dependency direction, canonical application symbols, non-monkey-patched answer authority, UI isolation from core layers, wildcard-import hygiene, required architecture-contract markers, deterministic dependency-lock presence, and Python syntax across project modules. CI runs this gate in every diagnostic, high-level, and regression lane.

This is deliberately a contract gate rather than a generic style linter: it protects boundaries that must survive refactors, while leaving implementation-level formatting and naming to ordinary tests and tooling.

## Scalability target

The reference deployment is a single machine with roughly 16GB RAM and a CPU-first workload. Concurrency is deliberately bounded around expensive model operations. Visual embeddings remain optional so figure/caption search does not make the default laptop profile unusable.

## Engineering quality target

The project treats architecture as executable policy. The composition boundary, thin application entrypoint, UI security boundary, canonical answer authority, bounded runtime settings, dependency direction, and architecture gate are protected by automated contract tests. Changes to these boundaries must update the contract and its tests together.

## Architectural debt register

The legacy `RAGSystem` surface remains transitional. Runtime installers and `rag_project.composition.prepare_runtime` are the composition boundary. The deep PDF contract is installed from that boundary so parsing, structure, versioning, validation and retrieval policies are active for production entry points.

The next architectural migration target is ownership-based retirement of remaining legacy UI and `RAGSystem` surfaces; until then, their use is treated as compatibility surface rather than production authority.
