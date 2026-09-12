# Deep Diagnostic Test System — 17 Phases

The repository has one authoritative diagnostic system: `scripts/test_doctor.py`, backed by `rag_project.testing.runner.UnifiedDiagnosticEngine`.

## Execution modes

```text
python scripts/test_doctor.py --fast
python scripts/test_doctor.py --deep
python scripts/test_doctor.py --all --json artifacts/deep_diagnostic.json
python scripts/test_doctor.py --phase 1 5 9 12 13 17
```

The engine runs on `master` and does not create branches or modify production files while probing. Mutation checks use temporary source copies only.

## Phase contract

| # | Phase | Purpose | Evidence produced |
|---:|---|---|---|
| 1 | Architecture/test coverage map | Map code domains, tests, and marker coverage. | Module/test/domain inventory. |
| 2 | Ultra-fast health gate | Collection + fast/contract checks. | Collected-test count + contract output. |
| 3 | Dependency-aware diagnostic chain | Locate the first broken diagnostic contract. | Executed ingestion → chunking → context chain. |
| 4 | Input/transformation/output contracts | Verify cross-module public boundaries. | Input, transformation, and output contract booleans. |
| 5 | Cross-layer invariants | Verify identity conservation through the real chunk/storage path. | Vector/SQLite counts, IDs, metadata, index validation. |
| 6 | Information-loss analysis | Measure representation survival instead of counting source tokens. | Token survival + schema-field survival matrix. |
| 7 | Adversarial document laboratory | Exercise hostile layout/text/OCR/table/figure variants. | Per-case chunk counts, schema validity, latency. |
| 8 | Metamorphic stability | Execute harmless transformations against production text/chunk utilities. | Normalization, tokenization, lexical-score, and chunk invariants. |
| 9 | Retrieval microscope | Measure real local vector/lexical retrieval and filtering. | Recall@3, MRR, semantic recall, metadata-filter correctness. |
| 10 | RAG answer causality | Trace retrieval → context → evidence identity. | Causal edge status and first broken edge. |
| 11 | Mutation detection | Execute non-destructive mutants against production utility contracts. | Applicable mutants, killed mutants, kill score. |
| 12 | Root-cause fingerprinting | Normalize observed failures into stable structured fingerprints. | Fingerprint groups, affected phases, confidence. |
| 13 | Failure-cascade compression | Collapse symptoms using phase/failure relationships. | First failed phase, affected phases, fix order. |
| 14 | Performance intelligence | Measure actual production-stage latency repeatedly. | p50/p95/p99 for normalization, chunking, lexical and semantic retrieval. |
| 15 | Resource/leak diagnostics | Measure bounded repeated production pipeline allocations. | Traced allocation series, growth signal, exercised stages. |
| 16 | Golden medical RAG benchmark | Execute the permanent gold set in offline retrieval mode or an opt-in live API mode. | Recall/MRR or live KPI measurements; clinical correctness is never inferred. |
| 17 | Master diagnostic certification | Verify 17/17 implementation and summarize runtime certification state. | Implementation matrix, missing phases, runtime failures/warnings/blockers. |

## Diagnostic principles

The system is designed around **information per second**, not test-count inflation.

1. A failed prerequisite is recorded as evidence, but downstream diagnostic probes are not suppressed merely because the upstream phase failed. A missing prerequisite result is the true blocking condition.
2. Every phase emits structured data: status, duration, evidence, score where meaningful, failure details, and blockers.
3. Phases 3–16 use bounded real project utilities rather than marker existence as their primary evidence path.
4. Root-cause grouping uses structured location, exception, normalized failure message, phase order, and affected-phase evidence.
5. Phase 17 distinguishes **implementation completeness (17/17)** from **runtime certification**. The system never claims a healthy runtime merely because all phase code exists.
6. Memory probes are bounded diagnostics, not 24-hour production certification. The existing long-duration runner remains separately available when a true duration certification is required.
7. The gold-set benchmark never claims clinical correctness merely because deterministic retrieval/path/term/citation checks pass.

## Reports

The JSON report is intended for machine consumption by CI, dashboards, and issue automation. CI now runs the authoritative full 17-phase diagnostic and explicitly verifies that phase 17 reports `implementation_coverage = 17/17`.
