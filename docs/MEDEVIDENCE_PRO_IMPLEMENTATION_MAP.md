# MedEvidence Pro implementation map

This branch implements the uploaded 20-week evidence-first plan against the existing production stack.

## Core cascade already present in master

- Safety gate and scope/emergency handling
- Intelligent query routing with intent, complexity, entities and path selection
- Tiered lexical/semantic/multi-query retrieval with early exit
- SQLite semantic retrieval cache
- Structured knowledge lookup hooks
- Evidence compilation, ranking, compression, numeric slot extraction and contradiction checks
- Path A extractive, Path B template, Path C constrained local LLM, Hybrid fallback and Path D abstention
- Active verification, grounding and numeric mismatch checks
- Citation-aware response formatting and confidence/disclaimer output
- Local feedback logging

## Production-completion additions

`rag_project/intelligence/production_ops.py` adds:

- query/answer/verification/feedback SQLite contract
- latency/error/path metrics with P50/P95/P99 calculations
- alert thresholds for latency and negative-feedback rates
- deterministic A/B assignment and variant comparison
- quarterly failure-analysis/retraining manifests
- daily/weekly-style local backup rotation
- retry with exponential backoff and circuit breaker
- concurrent benchmark helper with throughput and percentile reporting

`rag_project/knowledge/medical_kb.py` adds the local structured-KB schema for drugs, interactions, guidelines, contraindications and the disease graph. It intentionally does not bundle licensed third-party medical datasets.

`rag_project/api/med_evidence_api.py` provides the optional FastAPI contract for `/query`, `/feedback`, `/health` and `/metrics` without making FastAPI a mandatory dependency of the offline runtime.

## Operations

- `scripts/setup_medevidence_db.py` initializes the structured KB.
- `scripts/import_medevidence_kb.py` imports CSV/JSON exports into the selected KB table.
- `scripts/run_medevidence_ops.py` produces metrics/alerts, builds retraining manifests, or rotates backups.
- `scripts/run_medevidence_benchmarks.py` runs the portable concurrency/throughput harness.
- `requirements.medevidence.in` lists optional HTTP/NLP/monitoring dependencies from the plan without destabilizing the existing deterministic lockfile.

## Data boundary

The plan names large external datasets (for example thousands of drugs/interactions and guideline facts). This repository now contains the schemas/import path, validation surface and maintenance hooks; populating those tables requires the user's chosen public or properly licensed source exports.

## Validation

`tests/test_medevidence_plan_completion.py` covers the added operational contracts, KB schema initialization, A/B determinism, retraining manifest creation, backups, retry/circuit-breaker behavior and benchmark percentile reporting.
