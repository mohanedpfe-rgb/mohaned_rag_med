# BookRAG Medical high-level test suite

The primary project-specific acceptance suite lives under `tests/high_level/`.

## Execution

Fast local gate (target: <= 60 seconds on a normal multi-core development machine):

```bash
python -m pip install pytest-xdist
python -m pytest -q tests/high_level -m "high_level and not slow" -n auto --dist loadscope --durations=20
```

Full high-level suite, including OCR and large-document scenarios:

```bash
python -m pytest -q tests/high_level -m high_level -n auto --dist loadscope --durations=20
```

Optional real-Ollama coverage:

```bash
python -m pytest -q tests/high_level -m "high_level and requires_ollama"
```

The fast gate deliberately excludes tests marked `slow`. Those scenarios remain part of the complete high-level suite and the nightly CI gate, so the performance optimization does not remove coverage.

The shared high-level system and deterministic PDF fixtures are created once per pytest session/worker. An autouse isolation fixture clears conversation memory and restores the default LLM before each test so the shared runtime does not leak conversational or model-spy state between cases.

The suite keeps deterministic fixtures inside pytest temporary directories. It never downloads medical content during a test run. The controlled PDF generator is intentionally small and deterministic; a real OCR fixture pack can be added later without changing the behavior contracts.

## CI policy

`high-level-behavior` is the primary PR/push gate and runs the parallel `high_level and not slow` suite with `--dist loadscope` under a hard 60-second wall-clock budget. `regression-suite` keeps the previous contract/adversarial coverage as a secondary safety net. `high-level-nightly.yml` runs the complete high-level suite in parallel, the authoritative 13-phase plan, and optional real-Ollama tests when Ollama is actually available.

The suite is behavior-first: tests assert observable status, evidence/grounding, citations, path selection, latency budgets, isolation, multilingual behavior, resilience, security boundaries, and complete end-to-end scenarios rather than internal implementation details alone.