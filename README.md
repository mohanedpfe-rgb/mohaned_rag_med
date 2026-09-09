<div align="center">

# 📚 BookRAG Medical

### A local, citation-first PDF RAG assistant built for real documents.

<p>
  <a href="https://github.com/mohanedpfe-rgb/mohaned_rag_med/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mohanedpfe-rgb/mohaned_rag_med/ci.yml?branch=master&label=CI&logo=github" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Ollama-local%20AI-111827" alt="Ollama">
  <img src="https://img.shields.io/badge/Chroma-vector%20search-16A34A" alt="Chroma">
  <img src="https://img.shields.io/badge/License-see%20repo-64748B" alt="License">
</p>

<p>
  <b>Upload a PDF.</b> <b>Index it.</b> <b>Ask a question.</b> <b>See the evidence.</b>
</p>

</div>

---

> [!IMPORTANT]
> **🔐 ADMIN PASSWORD REQUIRED:** set `BOOKRAG_ADMIN_PASSWORD` in your local `.env` file.
>
> BookRAG has **no built-in/default administrator password**. The configured password must be at least 12 characters long.
> Copy `.env.example` to `.env` and replace the placeholder with a unique random secret before starting Streamlit.

> [!CAUTION]
> **This project is a document research tool, not a doctor.** Medical answers can be incomplete or wrong. Always verify important clinical information against the original source and qualified medical guidance.

## 🌟 What is BookRAG?

BookRAG lets you talk to your own PDF documents.

Instead of asking a general AI model to guess from memory, BookRAG first finds relevant passages inside your indexed documents and then asks a local language model to answer using that evidence.

The basic idea:

```text
PDFs
  ↓
Extract text
  ↓
Split into useful chunks
  ↓
Create embeddings
  ↓
Store vectors + lexical index
  ↓
Search
  ↓
Rerank the best passages
  ↓
Build a small evidence context
  ↓
Generate an answer with citations
```

The system is designed to be **local, inspectable, resilient, and friendly to a normal i5 / 16 GB RAM machine**.

---

## 🎯 What this project is good for

BookRAG is especially useful for:

- medical books and lecture notes
- research papers and technical PDFs
- university course material
- long manuals and reference documents
- collections where you need to know **where an answer came from**

It supports normal text PDFs and can use OCR for scanned/image-based pages when OCR is enabled.

---

## 🧠 Main technologies

| Part | Technology | Job |
|---|---|---|
| Interface | Streamlit | Simple visual application |
| PDF text extraction | PyMuPDF + pymupdf4llm | Read PDF pages and structure |
| OCR | RapidOCR | Read scanned/image pages when enabled |
| Embeddings | `nomic-embed-text` through Ollama | Turn text into vectors |
| Vector search | Chroma | Find semantically similar chunks |
| Lexical search | BM25 | Find exact words and phrases |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reorder the best candidates |
| Generation | `llama3.2:3b` through Ollama | Write the final answer |
| Durable state | SQLite | Track documents, pages, stages, and recovery state |
| Tests | pytest | Protect important behavior |

---

## ✨ Why the architecture is stronger than a basic RAG demo

A basic RAG demo often looks like this:

```text
PDF → chunks → embeddings → database → LLM
```

That is easy to build, but it becomes fragile when a PDF is huge, an extractor hangs, an embedding request fails, two workers process the same document, or an index is only half-written.

BookRAG adds production-style protections around those failure cases:

### 🛡️ Time-bounded extraction
Expensive PDF, table, OCR, and model operations are isolated and bounded so one bad page should not permanently freeze ingestion.

### 🔐 Version-aware indexing
A document version includes the important indexing identity, including parser/chunking settings, embedding profile, embedding dimension, and model identity. Changing the indexing contract creates a new version instead of silently mixing incompatible vectors.

### 🧱 BUILDING → READY visibility
New index data is built in an isolated state first. Incomplete work is not supposed to look like finished searchable data.

### 🔄 Recovery after failure
Failed semantic builds can clean their partial state so a retry starts from a known condition.

### 👑 Lease ownership
Long-running document jobs use ownership/lease checks so an old worker cannot safely overwrite the state of a newer worker after losing ownership.

### 🧠 Hybrid retrieval
Semantic search catches meaning. BM25 catches exact terminology. Using both is useful for technical and medical text where one changed word can matter.

### 🎯 Reranking
The first search pass finds candidates. A cross-encoder then reorders them before the final context is built.

### 🧾 Evidence-first generation
The generator is instructed to use retrieved evidence, preserve important units/negations, cite material claims, and abstain when the question is not supported.

### 🌍 Multilingual quality gates
Query checks and evidence handling are designed with English, French, and Arabic text in mind.

---

## 🖥️ The new Studio UI

The application now uses a dedicated functional studio console instead of making users debug backend internals manually.

### 🏠 Overview
See the real current state:

- number of documents
- ready documents
- documents still processing
- vector chunk count
- lexical chunk count

### 💬 Chat
Ask questions directly against indexed PDFs.

The UI shows:

- answer
- confidence
- answerability
- query quality
- citations
- retrieved evidence

### ⚙️ Health
Check the important runtime pieces:

- Ollama connectivity
- configured embedding model
- vector compatibility
- generation model
- models visible from Ollama

### ⏳ Ingestion
Watch persisted document state and recent process events.

You can refresh manually or enable live refresh while processing.

### 🧭 Background
See the real UI worker history and active document processing state.

### 🔎 Inspector
Pick a document and inspect:

- status
- page count
- embedding dimension
- version
- per-page extraction/OCR state
- index verification result
- vector metadata
- raw document state

### 🛠️ Settings
Change supported runtime settings without editing Python code.

---

## 🚀 Quick start for beginners

### 1. Install Python

Use **Python 3.11 or 3.12**.

Check your version:

```bash
python --version
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:
