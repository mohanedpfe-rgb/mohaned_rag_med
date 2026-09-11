# Intelligent Testing and Failure Localization

The project now has a layered test system designed to find the first broken subsystem before the expensive full suite runs.

## 1. One command: smart diagnosis

From the repository root:

```powershell
python scripts/test_doctor.py --smart
```

Smart mode runs, in order:

```text
1. static Python parse check
2. runtime mutation map
3. pytest collection health
4. runtime installer provenance
5. focused contract/storage probes
6. fast contract gate
7. optional full suite
```

For every failing layer it reports the test failure family and the first project-owned traceback frame, for example:

```text
[lexical-persistence] SQLite lexical mirror was not queryable after reopen
rag_project/storage/vector_store.py:412
```

The important distinction is that it reports both the **symptom** and the **owning source location**.

## 2. Runtime provenance

```powershell
python scripts/test_doctor.py --audit-runtime
```

This walks every runtime installer in a fresh process and tracks hot public symbols after each installer. It reports:

- baseline owner;
- every mutation point;
- first bad owner;
- final owner;
- source file and line;
- signature and runtime provenance flags.

This is specifically designed for the project's historical runtime monkeypatch/compatibility stack.

## 3. Fast focused probes

```powershell
python scripts/test_doctor.py --probe
```

The probe set covers the highest-risk contract families:

- follow-up rewriting;
- structured numeric diagnostics;
- answer-engine enhancer signature;
- document replacement/version retirement;
- lexical persistence after reopen;
- lexical repair against the authoritative vector record.

## 4. Parallel subsystem matrix

```powershell
python scripts/test_matrix.py
```

This launches several fast lanes with limited parallelism:

```text
contracts
runtime-diagnostics
storage
intelligence
ingestion
regression
```

A failed lane prints its marker expression and first project-owned traceback frame. This is the quickest way to see which subsystem is currently unhealthy.

## 5. Contract gate

```powershell
python scripts/test_doctor.py --contracts
```

or:

```powershell
pytest -q -m "fast and contract"
```

These tests validate public return shapes, signatures, cross-module contracts, and protocol boundaries.

## 6. Full suite

Only after the fast layers are clean:

```powershell
python scripts/test_doctor.py --full
```

or:

```powershell
pytest -q
```

The full suite remains the release gate. The diagnostic tools are the fast localization layer before that gate.

## 7. Automatic test categorisation

Tests no longer have to be manually marked one by one for the main subsystem groups. `tests/conftest.py` classifies tests from their path/name into categories such as:

- unit
- contract
- diagnostic
- regression
- ingestion
- retrieval
- intelligence
- generation
- storage
- integration
- slow
- requires_ollama

This prevents new tests from silently disappearing from subsystem checks.

## Recommended workflow

```text
New failure
   ↓
python scripts/test_doctor.py --smart
   ↓
first failing subsystem + first project source frame
   ↓
python scripts/test_doctor.py --audit-runtime   (when runtime mutation is involved)
   ↓
fix the owning production layer
   ↓
python scripts/test_matrix.py
   ↓
python scripts/test_doctor.py --full
```

Do not create another runtime compatibility patch until the provenance audit proves that the current owner is the correct place to change.
