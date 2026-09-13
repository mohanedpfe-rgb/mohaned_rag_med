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
ANSWER_SERVICE = ROOT / "rag_project" / "application_answer_service.py"
RUNTIME = ROOT / "rag_project" / "runtime.py"
BOOTSTRAP_STATE = ROOT / "rag_project" / "runtime_bootstrap_state.py"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"
LOCKFILE = ROOT / "requirements.lock"

MAX_APP_LINES = 55
REQUIRED_APPLICATION_SYMBOLS = {
    "MedEvidenceProductionRAGSystem",
    "create_rag_system",
    "runtime_contract",
}
REQUIRED_ANSWER_SERVICE_SYMBOLS = {
    "answer",
    "detect_answer_language",
    "install_runtime_adapters",
    "normalize_public_answer_path",
}
REQUIRED_COMPOSITION_SYMBOLS = {
    "prepare_runtime",
    "install_production_contracts",
    "normalize_runtime_environment",
    "runtime_is_prepared",
}
FORBIDDEN_COMPOSITION_IMPORT_PREFIXES = ("streamlit", "rag_project.app")
FORBIDDEN_APPLICATION_IMPORTS = {"rag_project.composition"}
FORBIDDEN_APPLICATION_INSTALLER_MODULES = {
    "rag_project.intelligence.pipeline_integrity",
    "rag_project.intelligence.production_contract_v2",
    "rag_project.ingestion.ingestion_contract",
    "rag_project.canonical_runtime",
}
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


def _imported_names(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported.add((node.module, alias.name))
    return imported


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

    app_lines = len(APP.read_text(encoding="utf-8").splitlines())
    if app_lines > MAX_APP_LINES:
        violations.append(f"app.py is {app_lines} lines; limit is {MAX_APP_LINES}")
    app_imports = _imports(APP)
    if "rag_project.application" in app_imports:
        violations.append("app.py must not own application-service imports; use the composition/UI boundary")

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

    missing_application = sorted(REQUIRED_APPLICATION_SYMBOLS - _symbols(APPLICATION))
    if missing_application:
        violations.append(f"application.py missing stable symbols: {missing_application}")
    answer_symbols = sorted(REQUIRED_ANSWER_SERVICE_SYMBOLS - _symbols(ANSWER_SERVICE))
    if answer_symbols:
        violations.append(f"application_answer_service.py missing stable symbols: {answer_symbols}")

    application_imports = _imports(APPLICATION)
    reverse_dependency = sorted(application_imports & FORBIDDEN_APPLICATION_IMPORTS)
    if reverse_dependency:
        violations.append(f"application.py imports composition root: {reverse_dependency}")
    if "rag_project.runtime" not in application_imports:
        violations.append("application.py must consume neutral runtime policy")
    if "rag_project.runtime_bootstrap_state" not in application_imports:
        violations.append("application.py must consume neutral prepared-runtime state")
    if "rag_project.application_answer_service" not in application_imports:
        violations.append("application.py must delegate answer behavior to application_answer_service")
    duplicated_installers = sorted(
        module
        for module in FORBIDDEN_APPLICATION_INSTALLER_MODULES
        if (module, "install") in _imported_names(APPLICATION)
    )
    if duplicated_installers:
        violations.append(f"application.py bypasses neutral installer ownership: {duplicated_installers}")
    application_source = APPLICATION.read_text(encoding="utf-8")
    if "MedEvidenceProductionRAGSystem" not in application_source:
        violations.append("application.py lost the canonical production service")
    if '"answer_monkey_patch": False' not in application_source:
        violations.append("canonical answer path must remain explicitly non-monkey-patched")
    if "_med_evidence_answer" not in application_source or "_certified_god_answer" not in application_source:
        violations.append("application.py must preserve the canonical answer binding compatibility seam")
    if "runtime_is_prepared()" not in application_source:
        violations.append("application factory must honor the prepared-runtime marker")

    runtime_symbols = _symbols(RUNTIME)
    if "install_application_contracts" not in runtime_symbols:
        violations.append("runtime.py must own the neutral application-contract bootstrap")
    if "rag_project.runtime" not in composition_imports:
        violations.append("composition.py must delegate contract installation to neutral runtime policy")
    bootstrap_imports = _imports(BOOTSTRAP_STATE)
    if any(name == "streamlit" or name.startswith("rag_project.app") for name in bootstrap_imports):
        violations.append("runtime_bootstrap_state.py must remain presentation-independent")

    answer_service_imports = _imports(ANSWER_SERVICE)
    if "streamlit" in answer_service_imports or any(name.startswith("rag_project.app") for name in answer_service_imports):
        violations.append("application_answer_service.py must remain presentation-independent")

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

    for path in _python_files():
        try:
            if _has_wildcard_import(path):
                violations.append(f"{path.relative_to(ROOT)} uses a wildcard import")
        except SyntaxError:
            continue

    architecture_text = ARCHITECTURE.read_text(encoding="utf-8")
    for marker in (
        "rag_project.composition.prepare_runtime",
        "rag_project.app.ui_security_boundary",
        "scripts/architecture_gate.py",
        "application_answer_service",
        "BOOKRAG_RUNTIME_PREPARED_VERSION",
        "runtime_bootstrap_state",
        "immutable published document versions",
        "app.py",
    ):
        if marker.lower() not in architecture_text.lower():
            violations.append(f"ARCHITECTURE.md is missing contract marker: {marker}")
    if not LOCKFILE.is_file() or not LOCKFILE.read_text(encoding="utf-8").strip():
        violations.append("requirements.lock must exist and be non-empty")

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
        "contract": "architecture-gate-v5",
    }


def main() -> int:
    report = inspect()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
