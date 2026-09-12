"""Fail-closed inventory check for the two-plan testing requirements."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

MIN_TOTAL_TEST_FUNCTIONS = 500
MIN_INTEGRATION_TEST_FUNCTIONS = 150
MIN_HIGH_LEVEL_TEST_FUNCTIONS = 80
TEST_NAME_PATTERN = re.compile(r"^test_[a-z0-9_]+__.+$")
EXPECTED_HIGH_LEVEL_PHASES = (
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
)


def _functions(path: Path) -> tuple[int, int, int, list[str]]:
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
    except (OSError, SyntaxError):
        return 0, 0, 0, []
    total = high_level = integration = 0
    invalid_names: list[str] = []
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
        if not TEST_NAME_PATTERN.match(node.name):
            invalid_names.append(f"{path}:{node.name}")
    return total, high_level, integration, invalid_names


def inspect(root: Path) -> dict[str, object]:
    tests = root / "tests"
    high_level_root = tests / "high_level"
    total = high_level = integration = 0
    files = 0
    invalid_names: list[str] = []
    phase_function_counts: dict[str, int] = {phase: 0 for phase in EXPECTED_HIGH_LEVEL_PHASES}
    missing_phases: list[str] = []
    unexpected_phase_dirs: list[str] = []

    for path in tests.rglob("test_*.py"):
        files += 1
        a, b, c, bad_names = _functions(path)
        total += a
        high_level += b
        integration += c
        invalid_names.extend(bad_names)
        try:
            relative = path.relative_to(high_level_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in phase_function_counts:
            phase_function_counts[relative.parts[0]] += b

    if high_level_root.is_dir():
        unexpected_phase_dirs = sorted(
            child.name
            for child in high_level_root.iterdir()
            if child.is_dir() and child.name not in EXPECTED_HIGH_LEVEL_PHASES
        )

    for phase in EXPECTED_HIGH_LEVEL_PHASES:
        phase_dir = high_level_root / phase
        if not phase_dir.is_dir() or phase_function_counts.get(phase, 0) <= 0:
            missing_phases.append(phase)

    effective_integration = integration + high_level
    result: dict[str, object] = {
        "test_files": files,
        "total_test_functions": total,
        "high_level_test_functions": high_level,
        "explicit_integration_test_functions": integration,
        "effective_integration_test_functions": effective_integration,
        "high_level_phase_count": len(EXPECTED_HIGH_LEVEL_PHASES) - len(missing_phases),
        "expected_high_level_phase_count": len(EXPECTED_HIGH_LEVEL_PHASES),
        "phase_function_counts": phase_function_counts,
        "missing_high_level_phases": missing_phases,
        "unexpected_high_level_phase_directories": unexpected_phase_dirs,
        "invalid_test_names": sorted(invalid_names),
        "meets_total_500": total >= MIN_TOTAL_TEST_FUNCTIONS,
        "meets_integration_150": effective_integration >= MIN_INTEGRATION_TEST_FUNCTIONS,
        "meets_high_level_floor": high_level >= MIN_HIGH_LEVEL_TEST_FUNCTIONS,
        "meets_13_phase_plan": not missing_phases and not unexpected_phase_dirs,
        "meets_test_naming_contract": not invalid_names,
    }
    result["ready"] = bool(
        result["meets_total_500"]
        and result["meets_integration_150"]
        and result["meets_high_level_floor"]
        and result["meets_13_phase_plan"]
        and result["meets_test_naming_contract"]
    )
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = inspect(root)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
