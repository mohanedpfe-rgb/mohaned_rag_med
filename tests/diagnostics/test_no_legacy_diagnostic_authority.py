"""Prevent diagnostic tests from silently exercising superseded implementations."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "diagnostics"

FORBIDDEN_IMPORTS = (
    "rag_project.testing.strict_phases",
    "rag_project.testing.strict_v2",
    "rag_project.testing.robust_probes",
)

ALLOWED_COMPATIBILITY_TESTS = {
    # Compatibility tests may explicitly exercise legacy adapters, but they are
    # not allowed to exist in the authoritative 17-phase diagnostic directory.
}


def test_authoritative_diagnostic_tests_do_not_import_superseded_modules() -> None:
    violations: list[tuple[str, str]] = []
    self_name = Path(__file__).name
    for path in TEST_ROOT.glob("test_*.py"):
        # This contract test necessarily contains the forbidden module names in
        # its own policy table; scanning itself would be a self-match, not a
        # real diagnostic-test dependency.
        if path.name == self_name or path.name in ALLOWED_COMPATIBILITY_TESTS:
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_IMPORTS:
            if forbidden in text:
                violations.append((str(path.relative_to(ROOT)), forbidden))
    assert not violations, f"superseded diagnostic implementations imported by authoritative tests: {violations}"
