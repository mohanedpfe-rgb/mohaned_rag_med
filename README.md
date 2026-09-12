# BookRAG Medical

Local-first medical PDF RAG studio for searchable document libraries, grounded retrieval, citations, safety checks, and bounded local generation.

## Core pipeline

`PDF → parse/OCR → chunk → embed/index → retrieve → verify → answer`

The production composition root is `rag_project.application.create_rag_system`. The canonical answer authority is the MedEvidence Pro production path, with READY-only retrieval, evidence verification, citation validation, and bounded local Ollama generation.

## Start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

For CI's fully pinned environment:

```bash
python -m pip install --require-hashes -r requirements.lock
```

## Local models

The project is designed for local Ollama operation. Model names are configured through the project settings/environment rather than hard-coded into the application workflow.

## Production guarantees

- Ingestion publishes only verified `READY` documents.
- Failed or partial documents are invisible to search.
- Retrieval preserves document identity and source metadata.
- Simple factual questions can use an extractive path without the LLM.
- Constrained generation uses retrieved evidence only.
- Grounding, citations, numeric consistency, contradiction handling, and safety are verified before public success.
- Unsupported or unsafe questions abstain instead of fabricating evidence.
- Runtime recovery uses a verified extractive fallback.
- Conversation memory stores successful useful answers only.

## Testing

The repository contains a strict 13-phase high-level behavior suite plus the broader regression suite. CI includes the strict diagnostic gate, high-level behavior gate, regression suite, and dependency security audit.

Run the high-level suite locally with:

```bash
python scripts/validate_test_inventory.py
python -m pytest -q tests/high_level -m high_level
```

## Safety

BookRAG is a document research/study tool, not a doctor. Medical information can be incomplete or conflicting; verify important clinical information against original sources and qualified medical guidance.

Never commit passwords, API keys, private PDFs, or machine-specific secrets.