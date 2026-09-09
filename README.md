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
> **🔐 DEFAULT BOOKRAG ADMIN PASSWORD: `becheikh_mohaned_rag`**
>
> This is the permanent built-in administrator password and remains valid for local BookRAG use. An additional password may be configured with `BOOKRAG_ADMIN_PASSWORD`; that setting does **not** replace or disable the permanent default.

> [!CAUTION]
> **This project is a document research tool, not a doctor.** Medical answers can be incomplete or wrong. Always verify important clinical information against the original source and qualified medical guidance.

## 🌟 What is BookRAG?

BookRAG lets you talk to your own PDF documents.

Instead of asking a general AI model to guess from memory, BookRAG first finds relevant passages inside your indexed documents and then asks a local language model to answer using that evidence.

The basic idea is:

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
| PDF text extraction | PyMuPDF | Read PDF pages and structure |
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

The application uses a dedicated functional studio console instead of making users debug backend internals manually.

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

### ⚙️ System
Check the important runtime pieces:

- Ollama connectivity
- configured embedding model
- vector compatibility
- generation model
- models visible from Ollama

### ⏳ Live Processing
Watch persisted document state and recent process events.

The UI refreshes automatically while processing.

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

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For reproducible dependency installation, use `requirements.lock` when it is available.

### 4. Install Ollama

Install Ollama on your computer, start it, and make sure it answers locally.

Then pull the models used by the default profile:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

The exact model names can be changed from configuration if you prefer another local model.

### 5. Start the application

```bash
streamlit run app.py
```

Or:

```bash
python run_dev.py
```

### 6. 🔐 Log in

Use the permanent administrator password:

```text
becheikh_mohaned_rag
```

No additional environment variable is required for the default login to work.

For an additional local administrator credential, set `BOOKRAG_ADMIN_PASSWORD` in your private `.env`. It must be at least 12 characters and does not disable the permanent default.

### 7. Add your first PDF

1. Open **BookRAG Medical**.
2. Use **Add PDFs** in the sidebar.
3. Pick a PDF.
4. Uploading automatically starts ingestion.
5. Watch **Live Processing** until the document becomes **READY**.
6. Open **Ask BookRAG** and ask a question.
7. Open **Inspector** to see the pages and vectors behind the answer.

That is the complete basic workflow.

---

## 📁 Project layout

```text
mohaned_rag_med/
│
├── app.py                         # Main application entry point
├── run_dev.py                     # Developer launch entry point
├── requirements.txt               # Human-readable dependency constraints
├── requirements.in                # Dependency source constraints
├── requirements.lock              # Generated reproducible lockfile
├── .env.example                   # Configuration example
├── ARCHITECTURE.md                # Architecture rules and ownership
│
├── rag_project/
│   ├── application.py             # Composition root
│   ├── runtime.py                 # Runtime policy bootstrap
│   ├── app/
│   │   ├── bookrag_ui.py          # Active Streamlit studio UI
│   │   ├── production_rag.py      # Production RAG service wrapper
│   │   └── rag_system.py           # Core compatibility RAG implementation
│   ├── chunking/                  # Chunk creation
│   ├── citations/                 # Citation handling
│   ├── configuration/             # Settings and hardware profile
│   ├── embeddings/                # Embedding service
│   ├── evaluation/                # Evaluation tools
│   ├── generation/                # Local LLM client
│   ├── ingestion/                 # Document state + monitoring
│   ├── intelligence/              # Grounding and medical safety gates
│   ├── ocr/                       # OCR service
│   ├── parsing/                   # PDF parsing
│   ├── reranking/                 # Cross-encoder reranker
│   ├── retrieval/                 # Hybrid retrieval + context building
│   ├── storage/                   # Chroma + lexical storage
│   └── utils/                     # Shared helpers
│
├── tests/                         # Automated safety and regression tests
└── .github/workflows/             # CI + dependency lock automation
```

---

## 🏗️ Architecture in simple words

The most important rule is:

> **The UI should not know how the internals are assembled.**

The application enters through one composition root:

```text
app.py
   ↓
BookRAG UI
   ↓
application.create_rag_system()
   ↓
Production RAG service
   ↓
Retriever / reranker / embeddings / storage / generation
```

This makes it easier to replace a component later without rewriting the whole application.

The runtime hardening layer installs reliability policies before the main service is constructed, and the project keeps that bootstrap in one explicit place.

See `ARCHITECTURE.md` for the project-level rules.

---

## ⚙️ Default hardware profile

The project is tuned for a modest local computer rather than assuming a large GPU server.

The current profile targets roughly:

| Setting | Default direction |
|---|---|
| CPU profile | i5 / 16 GB RAM class |
| Embedding batch size | 16 |
| Ollama concurrency | 1 |
| Ingestion workers | 2 |
| Chunk size | 600 |
| Chunk overlap | 100 |
| Context budget | 3200 tokens |
| OCR | Disabled by default in the i5 profile |

The embedding service automatically backs the batch size down after timeouts and splits oversized requests rather than failing the whole ingestion job.

---

## 🔒 Safety and reliability notes

This repository contains several defensive behaviors that are easy to miss in a beginner project.

### Prompt-injection resistant evidence handling
Retrieved PDF text is treated as **document data**, not trusted instructions. The generation layer is explicitly told not to obey commands that appear inside evidence.

### No silent support for weak questions
Very short, malformed, or low-signal questions can be rejected before generation instead of producing a confident-looking guess.

### Encrypted / malformed PDFs
Documents that cannot be safely classified or opened are handled as failures instead of being treated as successfully indexed.

### Physical page identity is preserved
Printed page numbers and physical PDF page positions are kept conceptually separate so citations and checkpoints do not silently drift.

### Partial embedding failure
The ingestion path adapts embedding batch size after timeouts, handles oversized request responses, validates every returned vector, and cleans partial index data before a retry.

### Safe file replacement
Cross-filesystem moves use copy-to-temporary-file and replace semantics rather than assuming every filesystem behaves exactly like the local disk.

---

## 🧪 Testing

Run the full test suite with:

```bash
python -m pytest -q
```

For a lightweight compile check:

```bash
python -m compileall -q rag_project app.py run_dev.py
```

The CI workflow exercises the project on Python 3.11 and 3.12 with a low-resource/local-test configuration.

---

## 📊 Evaluation

The repository includes evaluation code and datasets under `data/eval/` and `rag_project/evaluation/`.

The goal is not simply:

> “Did the LLM say something reasonable?”

The more useful questions are:

- Did retrieval find the right evidence?
- Did reranking improve the candidate order?
- Is the answer supported?
- Are citations aligned with claims?
- Does the system abstain when evidence is weak?
- Does behavior remain stable across English, French, and Arabic queries?

For production work, **retrieval quality and evidence alignment matter as much as the final generated prose**.

---

## 🛠️ Configuration

Start from:

```text
.env.example
```

Important values include the Ollama base URL, embedding model, generation model, retrieval settings, chunking settings, embedding batch size, embedding timeout, and hardware profile.

The permanent administrator password is always:

```text
becheikh_mohaned_rag
```

An optional additional local administrator password can be supplied with `BOOKRAG_ADMIN_PASSWORD`. Keep machine-specific `.env` files out of Git.

---

## 🧹 Data management

The Studio sidebar contains a cleanup action for managed document data.

Use it carefully: cleanup is intentionally destructive because it removes managed indexing/application state.

Generated databases, caches, OCR artifacts, and local runtime state are ignored by Git where appropriate.

---

## 🔁 Dependency locking

The project keeps two layers:

```text
requirements.in  →  human-maintained dependency constraints
requirements.lock →  generated fully resolved lockfile
```

GitHub Actions is configured to regenerate the lockfile when dependency inputs change.

This gives you a readable source constraint file **and** a path toward reproducible installs.

---

## 🌱 Growing the project later

The current architecture is intentionally suitable for a single-machine or small-server deployment.

When the project grows, the natural next steps are:

1. move more hardening behavior from compatibility/runtime patches into the owning core services
2. introduce stronger end-to-end integration tests with representative PDFs
3. add persistent job orchestration if multiple application processes need to share ingestion workers
4. separate local storage from a production object/index backend when the dataset becomes large
5. add stronger automated retrieval/answer quality gates before deployment

These are scaling paths, not requirements for the basic local application.

---

## 🧑‍💻 Contributing

A good change should be:

```text
small → testable → observable → documented
```

Before opening a contribution, run:

```bash
python -m compileall -q rag_project app.py run_dev.py
python -m pytest -q
```

Keep generated data, model artifacts, local vector stores, and Python bytecode out of commits.

---

## 🏆 Project quality goals

The project is explicitly built around five engineering goals:

| Goal | What it means here |
|---|---|
| 🧱 Proper architecture | Clear ownership and one application composition path |
| ✨ Clean code | Small responsibilities, typed boundaries, understandable behavior |
| 📈 Scalable | Bounded work, isolated indexing, versioned data, safe concurrency |
| 🔧 Maintainable | Tests, configuration, documentation, observable state |
| 🕰️ Built for the long run | Recovery, compatibility checks, deterministic dependencies, migration-friendly design |

The target is not to make the code look impressive.

The target is to make it **easy to trust, easy to debug, and safe to improve**.

---

## 📌 Current status

**Project stage:** production-hardening / local production candidate.

The core architecture and runtime safety controls have been significantly strengthened, and the user interface is built around real persisted state and explicit working actions.

Before calling any release “production certified,” run the full CI suite and validate the application against the real PDFs and hardware you intend to use.

---

<div align="center">

### 🔐 Permanent default administrator password: `becheikh_mohaned_rag`

**Made for people who want their documents to stay in control.** 📚

**Local first · Evidence first · Simple to use · Built to grow**

</div>
