"""Static architecture and maintainability gate.

This gate intentionally uses only the Python standard library so it can run
before the test suite and remain useful even when optional runtime dependencies
are unavailable. It checks durable architectural contracts rather than judging
implementation style heuristically.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
COMPOSITION = ROOT / "rag_project" / "composition.py"
APPLICATION = ROOT / "rag_project" / "application.py"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"
LOCKFILE = ROOT / "requirements.lock"

MAX_APP_LINES = 55
REQUIRED_APPLICATION_SYMBOLS = {
    "MedEvidenceProductionRAGSystem",
    "create_rag_system",
    "runtime_contract",
}
REQUIRED_COMPOSITION_SYMBOLS = {
    "prepare_runtime",
    "install_production_contracts",
    "normalize_runtime_environment",
}
FORBIDDEN_COMPOSITION_IMPORT_PREFIXES = ("streamlit", "rag_project.app")
CORE_DIRS = ("configuration", "ingestion", "retrieval", "storage", "intelligence")


def _python_files() -> Iterable[Path]:
    yield from ROOT.glob("*.py")
    yield from (ROOT / "rag_project").rglob("*.py")
    yield from (ROOT / "scripts").rglob("*.py")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _has_wildcard_import(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    )


def inspect() -> dict[str, object]:
    violations: list[str] = []

    # The executable entrypoint must stay a presentation/orchestration shell.
    app_lines = len(APP.read_text(encoding="utf-8").splitlines())
    if app_lines > MAX_APP_LINES:
        violations.append(f"app.py is {app_lines} lines; limit is {MAX_APP_LINES}")
    app_imports = _imports(APP)
    if "rag_project.application" in app_imports:
        violations.append("app.py must not own application-service imports; use the composition/UI boundary")

    # The composition root may orchestrate infrastructure contracts, but must
    # remain independent from Streamlit and presentation modules.
    composition_imports = _imports(COMPOSITION)
    bad_composition_imports = sorted(
        name
        for name in composition_imports
        if name == "streamlit" or name.startswith(FORBIDDEN_COMPOSITION_IMPORT_PREFIXES[1])
    )
    if bad_composition_imports:
        violations.append(f"composition.py depends on presentation imports: {bad_composition_imports}")
    missing = sorted(REQUIRED_COMPOSITION_SYMBOLS - _symbols(COMPOSITION))
    if missing:
        violations.append(f"composition.py missing stable symbols: {missing}")

    # The canonical application service must expose stable entrypoints.
    missing_application = sorted(REQUIRED_APPLICATION_SYMBOLS - _symbols(APPLICATION))
    if missing_application:
        violations.append(f"application.py missing stable symbols: {missing_application}")
    application_source = APPLICATION.read_text(encoding="utf-8")
    if "MedEvidenceProductionRAGSystem" not in application_source:
        violations.append("application.py lost the canonical production service")
    if '"answer_monkey_patch": False' not in application_source:
        violations.append("canonical answer path must remain explicitly non-monkey-patched")

    # Lower layers must never depend on the Streamlit/presentation surface.
    for layer in CORE_DIRS:
        layer_root = ROOT / "rag_project" / layer
        if not layer_root.is_dir():
            continue
        for path in layer_root.rglob("*.py"):
            try:
                imports = _imports(path)
            except SyntaxError as exc:
                violations.append(f"{path.relative_to(ROOT)} has invalid Python: {exc.msg}")
                continue
            bad = sorted(name for name in imports if name == "streamlit" or name.startswith("rag_project.app"))
            if bad:
                violations.append(f"{path.relative_to(ROOT)} imports presentation layer: {bad}")

    # Wildcard imports make dependency ownership and static analysis ambiguous.
    for path in _python_files():
        try:
            if _has_wildcard_import(path):
                violations.append(f"{path.relative_to(ROOT)} uses a wildcard import")
        except SyntaxError:
            # Compile errors are reported separately below with the filename.
            continue

    # Documentation and deterministic dependency resolution are part of the
    # long-lived contract, not optional project hygiene.
    architecture_text = ARCHITECTURE.read_text(encoding="utf-8")
    for marker in (
        "rag_project.composition.prepare_runtime",
        "rag_project.app.ui_security_boundary",
        "app.py",
        "immutable published document versions",
    ):
        if marker.lower() not in architecture_text.lower():
            violations.append(f"ARCHITECTURE.md is missing contract marker: {marker}")
    if not LOCKFILE.is_file() or not LOCKFILE.read_text(encoding="utf-8").strip():
        violations.append("requirements.lock must exist and be non-empty")

    # Parse every project Python file here so CI catches architectural syntax
    # regressions before pytest starts importing the application graph.
    syntax_failures: list[str] = []
    for path in _python_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            syntax_failures.append(f"{path.relative_to(ROOT)}: {exc}")
    violations.extend(syntax_failures)

    return {
        "ready": not violations,
        "app_lines": app_lines,
        "python_file_count": sum(1 for _ in _python_files()),
        "checked_core_layers": list(CORE_DIRS),
        "violations": violations,
        "contract": "architecture-gate-v1",
    }


def main() -> int:
    report = inspect()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
