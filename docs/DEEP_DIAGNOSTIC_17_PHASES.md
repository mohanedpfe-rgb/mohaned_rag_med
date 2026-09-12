# Deep Diagnostic Test System — 17 Phases

The repository has one authoritative diagnostic system: `scripts/test_doctor.py`, backed by `rag_project.testing.runner.UnifiedDiagnosticEngine`.

## Execution modes

```text
python scripts/test_doctor.py --fast
python scripts/test_doctor.py --deep
python scripts/test_doctor.py --all --json artifacts/deep_diagnostic.json
python scripts/test_doctor.py --phase 1 5 9 12 13 17
```

The engine runs only on `master`; it does not create branches or modify production files while probing.

## Phase contract

| # | Phase | Purpose | Cost class |
|---:|---|---|---|
| 1 | Architecture/test coverage map | Map code domains, tests, and marker coverage. | FAST |
| 2 | Ultra-fast health gate | Collection + fast/contract checks. | FAST |
| 3 | Dependency-aware diagnostic chain | Locate the first broken diagnostic contract. | FAST/DEEP |
| 4 | Input/transformation/output contracts | Verify public cross-module boundaries. | FAST/DEEP |
| 5 | Cross-layer invariants | Check source/page/section identity conservation across ingestion, chunking, retrieval, and storage. | FAST |
| 6 | Information-loss analysis | Measure the weakest identity surface across canonical boundary modules. | FAST |
| 7 | Adversarial document laboratory | Exercise hostile PDF/OCR/layout/resilience suites. | DEEP |
| 8 | Metamorphic stability | Check semantic invariants under harmless query transformations. | FAST |
| 9 | Retrieval microscope | Exercise ranking/retrieval diagnostics. | DEEP |
| 10 | RAG answer causality | Trace retrieval → intelligence/generation behavior. | DEEP |
| 11 | Mutation detection | Maintain a non-destructive mutation inventory and compile shadow mutants. | FAST/DEEP |
| 12 | Root-cause fingerprinting | Normalize failures into repeatable fingerprints. | FAST |
| 13 | Failure-cascade compression | Collapse downstream symptoms into fix-order candidates. | FAST |
| 14 | Performance intelligence | Execute latency contract tests with a bounded timeout. | DEEP |
| 15 | Resource/leak diagnostics | Run a bounded RSS smoke measurement; never label it 24-hour certification. | DEEP |
| 16 | Golden medical RAG benchmark | Validate the permanent gold-set schema and case inventory; live KPI measurement remains explicit. | DEEP |
| 17 | Master diagnostic certification | Produce the complete report, root causes, cascade, and next-fix recommendation. | FAST |

## Diagnostic principles

The system is designed around **information per second**, not test-count inflation.

1. Upstream failures block downstream phases when those downstream results would mostly be symptoms.
2. Every phase emits structured data: status, duration, evidence, failure details, and blockers.
3. Root-cause grouping uses phase order, exception fingerprints, location, and subsystem vocabulary.
4. Downstream failures are compressed into a smaller actionable fix order.
5. Memory smoke results are explicitly marked as non-certifying; the existing 24-hour runner remains the source of truth for real certification.
6. The gold-set benchmark never claims clinical correctness merely because deterministic path/term/citation checks pass.

## Reports

The JSON report is intended for machine consumption by CI, future dashboards, and issue automation. The terminal renderer is intentionally compact so a developer can identify the first broken area without reading hundreds of test failures.
