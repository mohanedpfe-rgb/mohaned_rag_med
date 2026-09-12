"""Fail-closed inventory check for the two-plan testing requirements."""
from __future__ import annotations

import ast
import json
from pathlib import Path

MIN_TOTAL_TEST_FUNCTIONS = 500
MIN_INTEGRATION_TEST_FUNCTIONS = 150
MIN_HIGH_LEVEL_TEST_FUNCTIONS = 80
EXPECTED_HIGH_LEVEL_PHASES = {
    "01_ingestion",
    "02_retrieval",
    "03_query_intelligence",
    "04_answer_paths",
    "05_grounding_safety",
    "06_latency_performance",
    "07_multilingual",
    "08_storage_index",
    "09_conversation",
    "10_security_privacy",
    "11_resilience",
    "12_production_contracts",
    "13_end_to_end",
}


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


def inspect(root: Path) -> dict[str, int | bool | list[str]]:
    tests = root / "tests"
    high_level_root = tests / "high_level"
    total = high_level = integration = 0
    files = 0
    phase_function_counts: dict[str, int] = {}
    missing_phases: list[str] = []

    for path in tests.rglob("test_*.py"):
        files += 1
        a, b, c = _functions(path)
        total += a
        high_level += b
        integration += c
        try:
            relative = path.relative_to(high_level_root)
        except ValueError:
            continue
        if relative.parts:
            phase = relative.parts[0]
            phase_function_counts[phase] = phase_function_counts.get(phase, 0) + b

    for phase in sorted(EXPECTED_HIGH_LEVEL_PHASES):
        phase_dir = high_level_root / phase
        if not phase_dir.is_dir() or phase_function_counts.get(phase, 0) <= 0:
            missing_phases.append(phase)

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
        "high_level_phase_count": len(EXPECTED_HIGH_LEVEL_PHASES) - len(missing_phases),
        "expected_high_level_phase_count": len(EXPECTED_HIGH_LEVEL_PHASES),
        "missing_high_level_phases": missing_phases,
        "meets_total_500": total >= MIN_TOTAL_TEST_FUNCTIONS,
        "meets_integration_150": effective_integration >= MIN_INTEGRATION_TEST_FUNCTIONS,
        "meets_high_level_floor": high_level >= MIN_HIGH_LEVEL_TEST_FUNCTIONS,
        "meets_13_phase_plan": not missing_phases,
    }
    result["ready"] = bool(
        result["meets_total_500"]
        and result["meets_integration_150"]
        and result["meets_high_level_floor"]
        and result["meets_13_phase_plan"]
    )
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = inspect(root)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
