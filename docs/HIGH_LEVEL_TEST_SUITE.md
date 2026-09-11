# BookRAG Medical high-level test suite

The primary project-specific acceptance suite lives under `tests/high_level/`.

## Execution

Fast local gate:

```bash
python -m pytest -q tests/high_level -m high_level -m 'not slow'
```

Full high-level suite:

```bash
python -m pytest -q tests/high_level -m high_level
```

Optional real-Ollama coverage:

```bash
python -m pytest -q tests/high_level -m 'high_level and requires_ollama'
```

The suite keeps deterministic fixtures inside pytest temporary directories. It never downloads medical content during a test run. The controlled PDF generator is intentionally small and deterministic; a real OCR fixture pack can be added later without changing the behavior contracts.

## CI policy

`high-level-behavior` is the primary PR/push gate. `regression-suite` keeps the previous contract/adversarial coverage as a secondary safety net. `high-level-nightly.yml` executes slow scenarios and runs real-Ollama tests only when an Ollama service is actually available.

The suite is behavior-first: tests assert observable status, evidence/grounding, citations, path selection, latency budgets, isolation, multilingual behavior, resilience, security boundaries, and complete end-to-end scenarios rather than internal implementation details alone.