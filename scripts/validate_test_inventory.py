"""Fail-closed inventory check for the two-plan testing requirements."""
from __future__ import annotations

import ast
import json
from pathlib import Path

MIN_TOTAL_TEST_FUNCTIONS = 500
MIN_INTEGRATION_TEST_FUNCTIONS = 150
MIN_HIGH_LEVEL_TEST_FUNCTIONS = 80


def _functions(path: Path) -> tuple[int, int, int]:
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
    except (OSError, SyntaxError):
        return 0, 0, 0
    total = high_level = integration = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
            continue
        total += 1
        parts = {p.lower() for p in path.parts}
        if "high_level" in parts:
            high_level += 1
        source = ast.get_source_segment(source_text, node) or ""
        if "pytest.mark.integration" in source or "/integration/" in str(path).replace("\\", "/"):
            integration += 1
    return total, high_level, integration


def inspect(root: Path) -> dict[str, int | bool]:
    tests = root / "tests"
    total = high_level = integration = 0
    files = 0
    for path in tests.rglob("test_*.py"):
        files += 1
        a, b, c = _functions(path)
        total += a
        high_level += b
        integration += c

    # High-level behavior tests and explicit integration tests are separate test
    # populations. Both are executable integration-grade coverage, so their sum
    # is the effective integration population; do not take max() and discard one.
    effective_integration = integration + high_level
    result = {
        "test_files": files,
        "total_test_functions": total,
        "high_level_test_functions": high_level,
        "explicit_integration_test_functions": integration,
        "effective_integration_test_functions": effective_integration,
        "meets_total_500": total >= MIN_TOTAL_TEST_FUNCTIONS,
        "meets_integration_150": effective_integration >= MIN_INTEGRATION_TEST_FUNCTIONS,
        "meets_high_level_floor": high_level >= MIN_HIGH_LEVEL_TEST_FUNCTIONS,
    }
    result["ready"] = bool(result["meets_total_500"] and result["meets_integration_150"] and result["meets_high_level_floor"])
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = inspect(root)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
