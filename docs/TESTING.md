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

The default local gate has a hard 120-second wall-clock budget. It covers deterministic unit, contract, regression, storage, ingestion, retrieval, intelligence, and generation tests that do not require an external service or explicitly slow scenarios.

The runner reports actual pytest failures separately from timeouts and terminates Windows worker process trees when a timeout occurs.

Advanced form:

```powershell
python scripts/test_full_fast.py --workers 4 --timeout 110 --budget 120
```

## 2. Complete release suite

The complete test inventory is intentionally not forced into the local two-minute budget. Run it explicitly with:

```powershell
python -m pytest -q --tb=short
```

The complete suite includes slow, integration, and external-service tests and remains the release gate.

## 3. Smart diagnosis

Before an expensive run:

```powershell
python scripts/test_doctor.py --smart
```

Smart mode runs inexpensive diagnostic layers first, then focused contract/storage checks and the fast contract gate.

## 4. Runtime provenance

```powershell
python scripts/test_doctor.py --audit-runtime
```

This walks every runtime installer in a fresh process and tracks hot public symbols after each installer. It reports baseline owner, mutation points, first bad owner, final owner, source file and line, signature, and runtime provenance flags.

## 5. Focused probes

```powershell
python scripts/test_doctor.py --probe
```

These cover follow-up rewriting, structured numeric diagnostics, answer-engine enhancer signatures, document replacement/version retirement, lexical persistence after reopen, and lexical repair.

## 6. Parallel subsystem matrix

```powershell
python scripts/test_matrix.py
```

This launches fast lanes for contracts, runtime diagnostics, storage, intelligence, ingestion, and regression and reports selected counts, failure families, failing node ids, tracebacks, and timing.

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
python -m pytest -q --tb=short   (release/CI validation)
```
