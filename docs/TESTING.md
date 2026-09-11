# Intelligent Testing and Failure Localization

The project uses a layered test strategy so local development can get a reliable answer quickly without confusing an intentionally expensive release suite with a two-minute gate.

## 1. Fast local gate

From the repository root:

```powershell
python scripts/test_doctor.py --full
```

This runs the deterministic local gate through `scripts/test_full_fast.py` using:

```text
not slow and not integration and not requires_ollama
```

The default local gate has a hard 120-second wall-clock budget. It is designed for deterministic unit, contract, regression, storage, ingestion, retrieval, intelligence, and generation coverage that does not require an external service or explicitly slow scenarios.

The runner reports actual pytest failures separately from timeouts and terminates Windows worker process trees when a timeout occurs, so timed-out children are not left running in the background.

Advanced form:

```powershell
python scripts/test_full_fast.py --workers 4 --timeout 110 --budget 120
```

## 2. Complete release suite

The complete test inventory is intentionally not forced into the local two-minute budget. Run it explicitly with:

```powershell
python scripts/test_doctor.py --release-full
```

or:

```powershell
python -m pytest -q --tb=short
```

The complete suite includes slow, integration, and external-service tests and remains the release gate.

## 3. Smart diagnosis

Before an expensive run:

```powershell
python scripts/test_doctor.py --smart
```

Smart mode runs the inexpensive diagnostic layers first and then the focused contract/storage checks and fast contract gate, reporting the first project-owned failure frame where possible.

## 4. Runtime provenance

```powershell
python scripts/test_doctor.py --audit-runtime
```

This walks every runtime installer in a fresh process and tracks hot public symbols after each installer. It reports baseline owner, every mutation point, first bad owner, final owner, source file and line, signature, and runtime provenance flags.

## 5. Fast focused probes

```powershell
python scripts/test_doctor.py --probe
```

The probe set covers follow-up rewriting, structured numeric diagnostics, answer-engine enhancer signatures, document replacement/version retirement, lexical persistence after reopen, and lexical repair against the authoritative vector record.

## 6. Parallel subsystem matrix

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

A failed lane prints its marker expression and first project-owned traceback frame.

## 7. Automatic test categorisation

`tests/conftest.py` classifies tests from their path/name into categories such as:

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
python scripts/test_doctor.py --probe
   ↓
python scripts/test_matrix.py
   ↓
python scripts/test_doctor.py --full
   ↓
python scripts/test_doctor.py --release-full   (release/CI validation)
```

Do not create another runtime compatibility patch until the provenance audit proves that the current owner is the correct place to change.
