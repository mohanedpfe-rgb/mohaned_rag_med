# Intelligent Testing and Failure Diagnosis

The project now has a fast diagnostic layer in addition to the full pytest suite.

## 1. Fastest command

From the repository root:

```powershell
python scripts/test_doctor.py
```

This performs two actions:

1. Audits the runtime installer stack and reports the first installer that mutates each hot public symbol.
2. Runs a small set of deterministic failure probes and groups recognized failures by root-cause class.

## 2. Runtime provenance only

```powershell
python scripts/test_doctor.py --audit-runtime
```

This is the most important command when a regression looks like a mysterious monkeypatch or compatibility failure. It reports:

- module and qualified symbol name;
- source file and line;
- current signature;
- `_runtime_*` provenance flags;
- first installer that changed the symbol.

The audit is executed in a fresh Python process and calls installers one by one, so it can identify the **first mutation point**, not merely the final broken state.

## 3. Fast reproduction probes

```powershell
python scripts/test_doctor.py --probe
```

The probe set intentionally contains one representative test for each high-risk contract family:

- follow-up rewriting;
- numeric diagnostics;
- answer-engine enhancer signature;
- document replacement;
- lexical persistence;
- repair consistency.

Use this before `pytest -q` when debugging a new regression.

## 4. Contract sentinels

```powershell
pytest -q -m "fast and contract"
```

These tests are intentionally small and deterministic. They validate public return shapes, function signatures, cross-module identity, and protocol-text boundaries.

## 5. Full suite

```powershell
python scripts/test_doctor.py --full
```

or:

```powershell
pytest -q
```

The full suite remains the release gate. The diagnostic layer is not a replacement for it; it is the fast path for finding the first broken contract before spending many minutes on the entire suite.

## Diagnostic rule

When a failure appears in the full suite:

```text
full pytest
   -> fast probe
   -> runtime provenance audit
   -> first mutating installer
   -> source-level fix
   -> contract sentinels
   -> full pytest
```

Do not add another runtime monkeypatch until the provenance audit identifies the exact owner of the behavior.
