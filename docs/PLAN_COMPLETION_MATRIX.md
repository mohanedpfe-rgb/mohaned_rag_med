# Two-plan completion matrix

This file is intentionally fail-closed. `IMPLEMENTED` means production code exists; `VALIDATED` means an executable test/benchmark or signed artifact proves the behavior; `BLOCKED-EXTERNAL` means the repository cannot honestly claim completion without required external/licensed medical data, real service execution, measured artifacts, or clinical review.

| Plan area | Implementation | Validation | Current status |
|---|---|---|---|
| Canonical MedEvidence Pro answer authority | `MedEvidenceProEngine.answer` wired from `rag_project/application.py` | High-level + regression suites | IMPLEMENTED / VALIDATION-GATED |
| Safety gate / abstention / emergency routing | `SafetyGate`, routing and verification stack | High-level grounding/safety suites | IMPLEMENTED / VALIDATION-GATED |
| Tier0 lexical + synonym retrieval | `MultiTierRetriever` | Retrieval/path tests | IMPLEMENTED / VALIDATION-GATED |
| True semantic cache | `SemanticRetrievalCache`: cosine >=0.95, 7-day TTL, 10k cap, 768-d production contract; wired at composition root | `tests/test_semantic_retrieval_cache.py` | IMPLEMENTED / VALIDATION-GATED |
| Dense Tier1 / multi-query Tier2 | Existing hybrid retriever + cascade orchestration | Retrieval/performance suites | IMPLEMENTED / VALIDATION-GATED |
| Evidence compiler / citation enforcement / grounding | Evidence compiler + evidence/final-answer guards | Grounding/safety suites | IMPLEMENTED / VALIDATION-GATED |
| Medical structured KB runtime | SQLite schema + import/runtime lookup | KB validation/readiness checks | IMPLEMENTED; data required |
| Required medical KB scale | Import/validation infrastructure | Count validation | BLOCKED-EXTERNAL until 10k drugs / 50k interactions / 100k guideline facts / 5k contraindications / 5k disease nodes are actually supplied |
| Clinical corpus scale | Corpus infrastructure and retrieval interfaces | Corpus manifest + ingestion validation | BLOCKED-EXTERNAL until licensed/public corpus is supplied and built |
| Constrained LLM path / hybrid fallback | Canonical cascade + resilient Ollama client | High-level + optional real-Ollama suite | IMPLEMENTED / REAL-SERVICE VALIDATION-GATED |
| Production telemetry | SQLite operations store + API + Prometheus | Ops/API tests | IMPLEMENTED / VALIDATION-GATED |
| Correct intent/path metrics | Strict `MetricsService` joined to feedback | Strict-operations tests | IMPLEMENTED |
| A/B statistical significance | In-process two-proportion z test + minimum sample rollout gate | Strict-operations tests | IMPLEMENTED |
| Backup integrity / SQLite online backup | Verified backup manager + integrity checks | Backup tests | IMPLEMENTED / RESTORE EVIDENCE REQUIRED |
| Executable retraining | Deterministic local multinomial NB trainer + `ExecutableRetrainingManager` | Retraining tests | IMPLEMENTED; real datasets required |
| High-level behavior suite | 13 required behavior domains | CI primary gate | IMPLEMENTED / CI-VALIDATION-GATED |
| Test inventory thresholds | `scripts/validate_test_inventory.py`; CI enforced | CI | IMPLEMENTED / CI-VALIDATION-GATED |
| 500+ tests / 150+ integration tests | Existing suite + fail-closed inventory checker | CI | VALIDATION-GATED; exact current count must come from CI run |
| >90% code coverage | Coverage artifact contract | Coverage run | BLOCKED until measured coverage artifact exists |
| 100 concurrent / 1000 QPM | Real HTTP load-test runner | Load-test artifact | BLOCKED until actually executed in an environment |
| 24h memory stability | Required artifact contract | Long-duration benchmark | BLOCKED until actually executed |
| KPI targets (accuracy, hallucination, grounding, latency) | Benchmark infrastructure/acceptance paths | KPI artifact | BLOCKED until real benchmark data exists |
| Clinical reviewer sign-off | Release-gate documentation | Signed artifact | BLOCKED-EXTERNAL |
| Human testing readiness | Fail-closed certification gate | Readiness audit | BLOCKED until every external artifact above exists |

## Release rule

The project must not report `ready=true` until every row requiring external validation has a real artifact and every implementation row has an executable test path. Placeholder files, synthetic medical data, or unexecuted benchmark scripts do not count as completion.
