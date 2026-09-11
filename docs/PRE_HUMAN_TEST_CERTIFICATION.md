# MedEvidence Pro — Pre-Human-Test Certification

Human testing is **blocked** until every item below is true.

## Software gate

Run:

```bash
python -m rag_project.quality.human_test_readiness --root .
```

The command must exit `0` and report `"ready": true`.

## Medical data gate

Populate the licensed/public knowledge sources through the importer, then run:

```bash
python scripts/validate_medevidence_kb.py --db data/medical_knowledge.sqlite3
```

Minimums required by the plan:

- 10,000 drugs
- 50,000 interactions
- 100,000 guideline facts
- 5,000 contraindication rules
- 5,000 disease-graph nodes/records

The gate also runs SQLite integrity validation.

## Automated validation

Before human testing, run:

```bash
python -m compileall -q rag_project app.py
python -m pytest -q
python scripts/run_medevidence_load_test.py --queries 100 --workers 16
```

GitHub Actions must pass on Python 3.11 and 3.12, including the security audit.

## Operational validation

The production stack must have:

- local-first execution with cloud disabled by default
- structured query/answer/verification/feedback storage
- backup/restore procedure tested on a copy
- alert and metrics endpoints available
- Prometheus-compatible metric export
- deterministic A/B assignment and statistical evaluation
- retraining/failure-analysis manifest generation
- retry and circuit-breaker controls
- deployment service definitions reviewed

## Clinical validation

Only after all automated gates pass:

1. Medical reviewer signs off on the benchmark set.
2. Safety and emergency-routing cases are manually reviewed.
3. Citation correctness is checked against the source documents.
4. Human testing starts with a controlled, limited cohort.

This file is a release gate, not a claim that the system has already achieved the plan's accuracy or safety KPIs.
