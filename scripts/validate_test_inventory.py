"""Fail-closed inventory check for the two-plan testing requirements."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

MIN_TOTAL_TEST_FUNCTIONS = 500
MIN_INTEGRATION_TEST_FUNCTIONS = 150
MIN_HIGH_LEVEL_TEST_FUNCTIONS = 80
MIN_GOLD_CASES = 20
GOLD_REQUIRED_FIELDS = {
    "id",
    "question",
    "language",
    "expected_status",
    "expected_path",
    "must_contain",
    "must_not_contain",
    "must_cite",
    "max_latency_s",
    "must_not_call_llm",
}
TEST_NAME_PATTERN = re.compile(r"^test_[a-z0-9_]+__.+$")
EXPECTED_HIGH_LEVEL_PHASES = (
    "01_ingestion", "02_retrieval", "03_query_intelligence", "04_answer_paths",
    "05_grounding_safety", "06_latency_performance", "07_multilingual", "08_storage_index",
    "09_conversation", "10_security_privacy", "11_resilience", "12_production_contracts", "13_end_to_end",
)


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _functions(path: Path) -> tuple[int, int, int, list[str], list[str]]:
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
    except (OSError, SyntaxError):
        return 0, 0, 0, [], []
    total = high_level = integration = 0
    invalid_names: list[str] = []
    missing_marks: list[str] = []
    is_high_level_file = "high_level" in {p.lower() for p in path.parts}
    is_integration_path = "/integration/" in str(path).replace("\\", "/")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
            continue
        total += 1
        decorators = {_decorator_name(item) for item in node.decorator_list}
        marked_high_level = "pytest.mark.high_level" in decorators or "high_level" in decorators
        marked_integration = "pytest.mark.integration" in decorators or "integration" in decorators
        if is_high_level_file:
            high_level += 1
            if not marked_high_level:
                missing_marks.append(f"{path}:{node.name}")
        if marked_integration or is_integration_path:
            integration += 1
        if not TEST_NAME_PATTERN.match(node.name):
            invalid_names.append(f"{path}:{node.name}")
    return total, high_level, integration, invalid_names, missing_marks


def _gold_contract(root: Path) -> dict[str, object]:
    path = root / "tests" / "support" / "gold_sets" / "core.jsonl"
    errors: list[str] = []
    cases: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(item, dict):
                errors.append(f"line {line_number}: case must be an object")
                continue
            missing = sorted(GOLD_REQUIRED_FIELDS - set(item))
            if missing:
                errors.append(f"line {line_number}: missing fields {missing}")
            cases.append(item)
    except OSError as exc:
        errors.append(f"cannot read gold set: {exc}")
    ids = [str(item.get("id", "")) for item in cases]
    duplicate_ids = sorted({value for value in ids if value and ids.count(value) > 1})
    if duplicate_ids:
        errors.append(f"duplicate gold case ids: {duplicate_ids}")
    return {
        "path": str(path),
        "gold_case_count": len(cases),
        "meets_gold_floor": len(cases) >= MIN_GOLD_CASES,
        "gold_contract_errors": errors,
        "gold_unique_ids": not duplicate_ids,
        "ready": len(cases) >= MIN_GOLD_CASES and not errors,
    }


def inspect(root: Path) -> dict[str, object]:
    tests = root / "tests"
    high_level_root = tests / "high_level"
    total = high_level = integration = files = 0
    invalid_names: list[str] = []
    missing_marks: list[str] = []
    phase_function_counts = {phase: 0 for phase in EXPECTED_HIGH_LEVEL_PHASES}
    missing_phases: list[str] = []
    unexpected_phase_dirs: list[str] = []

    for path in tests.rglob("test_*.py"):
        files += 1
        a, b, c, bad_names, bad_marks = _functions(path)
        total += a; high_level += b; integration += c
        invalid_names.extend(bad_names); missing_marks.extend(bad_marks)
        try:
            relative = path.relative_to(high_level_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in phase_function_counts:
            phase_function_counts[relative.parts[0]] += b

    if high_level_root.is_dir():
        unexpected_phase_dirs = sorted(
            child.name for child in high_level_root.iterdir()
            if child.is_dir() and child.name not in EXPECTED_HIGH_LEVEL_PHASES
        )
    for phase in EXPECTED_HIGH_LEVEL_PHASES:
        phase_dir = high_level_root / phase
        if not phase_dir.is_dir() or phase_function_counts.get(phase, 0) <= 0:
            missing_phases.append(phase)

    effective_integration = integration + high_level
    gold = _gold_contract(root)
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
        "missing_high_level_marks": sorted(missing_marks),
        "meets_total_500": total >= MIN_TOTAL_TEST_FUNCTIONS,
        "meets_integration_150": effective_integration >= MIN_INTEGRATION_TEST_FUNCTIONS,
        "meets_high_level_floor": high_level >= MIN_HIGH_LEVEL_TEST_FUNCTIONS,
        "meets_13_phase_plan": not missing_phases and not unexpected_phase_dirs,
        "meets_test_naming_contract": not invalid_names,
        "meets_high_level_marker_contract": not missing_marks,
        "gold_set": gold,
        "meets_gold_contract": bool(gold["ready"]),
    }
    result["ready"] = bool(
        result["meets_total_500"] and result["meets_integration_150"] and result["meets_high_level_floor"]
        and result["meets_13_phase_plan"] and result["meets_test_naming_contract"]
        and result["meets_high_level_marker_contract"] and result["meets_gold_contract"]
    )
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = inspect(root)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
