# MedEvidence Pro

MedEvidence Pro is now the active production answer stack for the repository. It replaces the legacy answer-generation path while preserving the existing ingestion, vector storage, citation, security, and UI service shell.

## Runtime architecture

The active request path is:

`Safety Gate -> Query Router -> Tiered Retrieval -> Structured Knowledge -> Evidence Compiler -> Answer Cascade -> Active Verification -> Response Formatter -> Feedback Logger`

The local `llama3.2:3b` model is used only for constrained synthesis when extractive or template paths are insufficient. All generated answers must remain grounded in retrieved evidence and source markers.

## Retrieval tiers

- **Tier 0:** lexical retrieval with medical synonym expansion and early exit.
- **Tier 1:** existing hybrid dense + lexical retrieval over query variants.
- **Tier 2:** parallel multi-query retrieval for high-complexity / multi-hop questions.
- **Cache:** SQLite retrieval cache with a seven-day TTL.

## Structured knowledge

An optional local SQLite database at `data/medical_knowledge.sqlite3` is recognized when present. The engine looks for compatible tables named `drugs`, `interactions`, `contraindications`, and `guidelines`. Missing tables are treated as an empty knowledge layer, not as a runtime failure.

## Safety and verification

The safety layer is deterministic and local: harmful/illicit request patterns, medical-scope checks, emergency signals, real-patient language, and high-rigor contexts are evaluated before generation. Verification uses the repository's existing evidence-guard and final-answer contracts plus numeric consistency checks.

Emergency signals are flagged rather than silently answered as routine education. The response is still grounded in indexed evidence and carries an urgent-care disclaimer.

## Feedback

Answer telemetry is stored locally in `data/med_evidence_feedback.sqlite3` with query, path, confidence, latency, status, and review-needed metadata. The feedback store is intentionally local so the default deployment remains serverless/offline-friendly.

## Compatibility

`ProductionRAGSystem` remains the ingestion/storage/security service shell. The application composition root now instantiates `MedEvidenceProductionRAGSystem`, whose answer authority is:

`rag_project.intelligence.med_evidence_pro.MedEvidenceProEngine.answer`

Existing legacy telemetry keys such as `god_mode_100` are retained as compatibility indicators only; they no longer identify the implementation authority.

## Validation

CI on `master` continues to compile the complete application and run the full pytest suite on Python 3.11 and 3.12. New MedEvidence Pro unit/contract tests cover safety, routing, caching, extraction, grounding, and abstention behavior.
